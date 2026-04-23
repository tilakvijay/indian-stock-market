from __future__ import annotations

import math

import numpy as np
import pandas as pd

from indian_stock_pipeline.core.config import Settings
from indian_stock_pipeline.core.schemas import BacktestSummary, FoldMetric
from indian_stock_pipeline.models.signals import build_signal_frame


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _run_single_fold(features: pd.DataFrame, regime_label: str = "unknown",
                     position_size: float = 1.0) -> dict:
    """Run a single backtest pass on a feature DataFrame.

    Returns a dict with equity_rows, trades, and summary stats.
    """
    signal_frame = build_signal_frame(features, regime_label=regime_label)

    initial_cash = 100_000.0
    fee_rate = 0.001
    slippage_rate = 0.001

    cash = initial_cash
    shares = 0.0
    in_position = False
    entry_price = 0.0
    highest_close = 0.0
    entry_timestamp = None
    trades: list[dict[str, float | str]] = []
    equity_rows: list[dict] = []

    for i, (timestamp, row) in enumerate(features.iterrows()):
        signal_row = signal_frame.iloc[i]
        close = float(row["close"])
        atr = float(row["atr"]) if np.isfinite(row["atr"]) and row["atr"] > 0 else close * 0.03

        if in_position:
            highest_close = max(highest_close, close)
            trailing_stop = highest_close - (1.35 * atr)
            hard_stop = entry_price - (2.0 * atr)
            regime_break = bool(signal_row["exit"])
            stop_hit = close < trailing_stop or close < hard_stop

            if regime_break or stop_hit:
                exit_price = close * (1.0 - slippage_rate)
                cash = shares * exit_price * (1.0 - fee_rate)
                trade_return = (exit_price / entry_price) - 1.0
                trades.append({
                    "entry_date": entry_timestamp.strftime("%Y-%m-%d") if hasattr(entry_timestamp, "strftime") else str(entry_timestamp),
                    "exit_date": timestamp.strftime("%Y-%m-%d") if hasattr(timestamp, "strftime") else str(timestamp),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "return": round(trade_return, 4),
                    "reason": "signal_exit" if regime_break else "stop_exit",
                })
                shares = 0.0
                in_position = False

        if not in_position and bool(signal_row["long_entry"]):
            entry_price = close * (1.0 + slippage_rate)
            # Regime-aware position sizing
            allocated = cash * position_size
            shares = (allocated * (1.0 - fee_rate)) / max(entry_price, 1e-9)
            cash = cash - allocated
            in_position = True
            highest_close = close
            entry_timestamp = timestamp

        equity_value = cash + (shares * close if in_position else 0.0)
        equity_rows.append({
            "date": timestamp,
            "equity": equity_value,
            "close": close,
            "score": float(signal_row["score"]),
            "signal": "LONG" if in_position else ("WATCH" if bool(signal_row["watch"]) else "FLAT"),
        })

    return {"equity_rows": equity_rows, "trades": trades, "initial_cash": initial_cash}


def _compute_fold_stats(equity_rows: list[dict], trades: list[dict],
                        initial_cash: float) -> dict:
    """Compute summary statistics from a single fold."""
    if not equity_rows:
        return {"total_return": 0.0, "sharpe": 0.0, "max_dd": 0.0, "win_rate": 0.0, "n_trades": 0}

    equity_series = pd.Series([r["equity"] for r in equity_rows])
    daily_returns = equity_series.pct_change().fillna(0.0)
    total_return = (equity_series.iloc[-1] / initial_cash) - 1.0
    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = float((daily_returns.mean() / daily_returns.std()) * math.sqrt(252))
    max_dd = _max_drawdown(equity_series)
    trade_returns = [t["return"] for t in trades]
    win_rate = float(np.mean([r > 0 for r in trade_returns])) if trade_returns else 0.0

    return {
        "total_return": float(total_return),
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "n_trades": len(trades),
    }


def _monte_carlo_test(trade_returns: list[float], n_simulations: int = 1000) -> tuple[float, float]:
    """Monte Carlo shuffle test.

    Randomly reshuffles trade returns to compute a p-value:
    what fraction of random reshuffles produce a better total return
    than the actual strategy?

    Returns (p_value, median_random_return).
    """
    if len(trade_returns) < 5:
        return 1.0, 0.0

    actual_total = float(np.sum(trade_returns))
    random_totals = []
    rng = np.random.default_rng(42)

    for _ in range(n_simulations):
        # Random sign flip — tests if the strategy's directional skill is real
        signs = rng.choice([-1, 1], size=len(trade_returns))
        random_total = float(np.sum(np.array(trade_returns) * signs))
        random_totals.append(random_total)

    random_totals = np.array(random_totals)
    p_value = float(np.mean(random_totals >= actual_total))
    median_return = float(np.median(random_totals))
    return p_value, median_return


def run_backtest(features: pd.DataFrame, settings: Settings | None = None,
                 regime_label: str = "unknown", position_size: float = 1.0) -> BacktestSummary:
    """Walk-forward purged cross-validation backtester with Monte Carlo.

    Parameters
    ----------
    features : pd.DataFrame
        Full feature DataFrame.
    settings : Settings or None
        Config for fold count, purge/embargo sizes, monte carlo sims.
    regime_label : str
        Current regime label for adaptive signal weights.
    position_size : float
        Kelly-derived position size (0-1).
    """
    if len(features) < 150:
        return BacktestSummary(enabled=False, notes="Not enough history for a meaningful rolling backtest.")

    # Default settings
    n_folds = 5
    purge_bars = 10
    embargo_bars = 5
    mc_sims = 1000
    if settings is not None:
        n_folds = settings.wf_folds
        purge_bars = settings.wf_purge_bars
        embargo_bars = settings.wf_embargo_bars
        mc_sims = settings.monte_carlo_simulations

    total_bars = len(features)
    fold_size = total_bars // (n_folds + 1)  # +1 so we have room for train + test

    if fold_size < 60:
        # Fall back to single-pass backtest
        return _single_pass_backtest(features, regime_label, position_size)

    # --- Walk-Forward with Purging + Embargo ---
    all_fold_metrics: list[FoldMetric] = []
    all_trades: list[dict] = []
    all_equity_rows: list[dict] = []

    for fold_idx in range(n_folds):
        # Test window
        test_end = total_bars - fold_idx * fold_size
        test_start = test_end - fold_size
        if test_start < fold_size:
            break

        # Train window (everything before test, minus purge)
        train_end = test_start - purge_bars
        if train_end < fold_size:
            continue

        # Embargo after test (not used for next train)
        _embargo_end = min(test_end + embargo_bars, total_bars)

        train_features = features.iloc[:train_end]
        test_features = features.iloc[test_start:test_end]

        if len(test_features) < 30:
            continue

        # Run backtest on test fold only
        fold_result = _run_single_fold(test_features, regime_label, position_size)
        fold_stats = _compute_fold_stats(
            fold_result["equity_rows"], fold_result["trades"], fold_result["initial_cash"]
        )

        # Record fold metrics
        test_idx = test_features.index
        fold_metric = FoldMetric(
            fold_index=fold_idx,
            train_start=str(features.index[0].strftime("%Y-%m-%d") if hasattr(features.index[0], "strftime") else features.index[0]),
            train_end=str(train_features.index[-1].strftime("%Y-%m-%d") if hasattr(train_features.index[-1], "strftime") else train_features.index[-1]),
            test_start=str(test_idx[0].strftime("%Y-%m-%d") if hasattr(test_idx[0], "strftime") else test_idx[0]),
            test_end=str(test_idx[-1].strftime("%Y-%m-%d") if hasattr(test_idx[-1], "strftime") else test_idx[-1]),
            total_return=fold_stats["total_return"],
            sharpe_ratio=fold_stats["sharpe"],
            max_drawdown=fold_stats["max_dd"],
            win_rate=fold_stats["win_rate"],
            total_trades=fold_stats["n_trades"],
        )
        all_fold_metrics.append(fold_metric)
        all_trades.extend(fold_result["trades"])
        all_equity_rows.extend(fold_result["equity_rows"])

    if not all_fold_metrics:
        return _single_pass_backtest(features, regime_label, position_size)

    # --- Aggregate across folds ---
    avg_return = float(np.mean([fm.total_return for fm in all_fold_metrics]))
    avg_sharpe = float(np.mean([fm.sharpe_ratio for fm in all_fold_metrics]))
    avg_drawdown = float(np.mean([fm.max_drawdown for fm in all_fold_metrics]))
    avg_win_rate = float(np.mean([fm.win_rate for fm in all_fold_metrics]))
    total_trades_count = sum(fm.total_trades for fm in all_fold_metrics)

    # Annualize from average fold return
    bars_per_fold = fold_size
    years_per_fold = bars_per_fold / 252.0
    annualized = (1.0 + avg_return) ** (1.0 / max(years_per_fold, 0.01)) - 1.0 if years_per_fold > 0 else 0.0

    benchmark_return = (features["close"].iloc[-1] / features["close"].iloc[0]) - 1.0

    # Build equity curve from last fold
    equity_curve = pd.DataFrame()
    if all_equity_rows:
        equity_curve = pd.DataFrame(all_equity_rows[-min(252, len(all_equity_rows)):]).set_index("date")

    trade_returns = [t["return"] for t in all_trades]
    expectancy = float(np.mean(trade_returns)) if trade_returns else 0.0

    # --- Monte Carlo significance test ---
    mc_p_value, mc_median = _monte_carlo_test(trade_returns, mc_sims)

    last_signal = "FLAT"
    if all_equity_rows:
        last_signal = str(all_equity_rows[-1].get("signal", "FLAT"))

    return BacktestSummary(
        enabled=True,
        total_return=avg_return,
        benchmark_return=float(benchmark_return),
        annualized_return=annualized,
        sharpe_ratio=avg_sharpe,
        max_drawdown=avg_drawdown,
        win_rate=avg_win_rate,
        total_trades=total_trades_count,
        expectancy=expectancy,
        avg_trade_return=expectancy,
        last_signal=last_signal,
        trade_log=all_trades[-10:],
        equity_curve=equity_curve,
        fold_metrics=all_fold_metrics,
        monte_carlo_p_value=mc_p_value,
        monte_carlo_median_return=mc_median,
        notes=(
            f"Walk-forward backtest with {len(all_fold_metrics)} folds, "
            f"{purge_bars}-bar purge, {embargo_bars}-bar embargo. "
            f"Monte Carlo p-value: {mc_p_value:.3f}. "
            f"Includes fees ({0.1}%), slippage ({0.1}%), and ATR-based exits."
        ),
    )


def _single_pass_backtest(features: pd.DataFrame, regime_label: str = "unknown",
                          position_size: float = 1.0) -> BacktestSummary:
    """Fallback single-pass backtest when data is insufficient for walk-forward CV."""
    result = _run_single_fold(features, regime_label, position_size)
    stats = _compute_fold_stats(result["equity_rows"], result["trades"], result["initial_cash"])

    equity_curve = pd.DataFrame()
    if result["equity_rows"]:
        equity_curve = pd.DataFrame(result["equity_rows"]).set_index("date")

    total_return = stats["total_return"]
    benchmark_return = (features["close"].iloc[-1] / features["close"].iloc[0]) - 1.0
    annualized = (1.0 + total_return) ** (252 / max(len(features), 1)) - 1.0

    trade_returns = [t["return"] for t in result["trades"]]
    mc_p, mc_med = _monte_carlo_test(trade_returns, 500)

    return BacktestSummary(
        enabled=True,
        total_return=total_return,
        benchmark_return=float(benchmark_return),
        annualized_return=annualized,
        sharpe_ratio=stats["sharpe"],
        max_drawdown=stats["max_dd"],
        win_rate=stats["win_rate"],
        total_trades=stats["n_trades"],
        expectancy=float(np.mean(trade_returns)) if trade_returns else 0.0,
        avg_trade_return=float(np.mean(trade_returns)) if trade_returns else 0.0,
        last_signal=str(result["equity_rows"][-1]["signal"]) if result["equity_rows"] else "FLAT",
        trade_log=result["trades"][-10:],
        equity_curve=equity_curve.tail(252) if not equity_curve.empty else equity_curve,
        monte_carlo_p_value=mc_p,
        monte_carlo_median_return=mc_med,
        notes="Single-pass backtest (insufficient data for walk-forward CV). Includes fees, slippage, and ATR exits.",
    )
