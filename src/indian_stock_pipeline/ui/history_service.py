"""Analysis history — persist and retrieve past stock analysis runs."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HISTORY_DIR = Path(".cache/analysis_history")


def _ensure_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _serialize_result(result) -> dict[str, Any]:
    """Convert an AnalysisResult to a JSON-safe dict (key fields only)."""
    rec = result.recommendation
    regime = result.regime
    latest = result.features.iloc[-1]

    entry: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "symbol": result.instrument.symbol,
        "display_name": result.instrument.display_name,
        "exchange": result.instrument.exchange,
        "action": rec.action,
        "confidence": round(rec.confidence, 4),
        "setup_quality": rec.setup_quality,
        "regime_label": rec.regime_label,
        "close": round(float(latest["close"]), 2),
        "entry_low": round(rec.entry_low, 2),
        "entry_high": round(rec.entry_high, 2),
        "stop_loss": round(rec.stop_loss, 2),
        "exit_target": round(rec.exit_target, 2),
        "risk_reward": round(rec.risk_reward, 2),
        "position_size": round(rec.position_size, 4),
        "kelly_fraction": round(rec.kelly_fraction, 4),
        "headline": rec.headline,
        "summary": rec.summary,
        "horizon": rec.horizon,
        "risk_flags": rec.risk_flags,
        "notes": rec.notes,
    }

    # Add key metrics
    for key in ["rsi_14", "adx_14", "momentum_20", "momentum_63", "hurst",
                "atr", "realized_vol", "delivery_pct", "put_call_ratio"]:
        val = latest.get(key)
        if val is not None:
            try:
                entry[key] = round(float(val), 4)
            except (TypeError, ValueError):
                pass

    # Regime info
    entry["vix_zscore"] = round(regime.vix_zscore, 2)
    entry["fii_dii_direction"] = regime.fii_dii_direction
    entry["trending_probability"] = round(regime.trending_probability, 4)

    # Backtest summary
    bt = result.backtest
    if bt and bt.enabled:
        entry["backtest"] = {
            "total_return": round(bt.total_return or 0, 4),
            "sharpe_ratio": round(bt.sharpe_ratio or 0, 2),
            "max_drawdown": round(bt.max_drawdown or 0, 4),
            "win_rate": round(bt.win_rate or 0, 4),
            "total_trades": bt.total_trades or 0,
            "mc_p_value": round(bt.monte_carlo_p_value or 1, 3),
        }

    # Best buy timing data
    entry["best_buy_timing"] = _compute_best_buy_timing(result.features)

    return entry


def _compute_best_buy_timing(features) -> dict[str, Any]:
    """Find the best buy points in the last year and estimate next opportunity."""
    import numpy as np

    timing: dict[str, Any] = {}
    try:
        close = features["close"].tail(252)  # ~1 year
        if len(close) < 30:
            return {"note": "Insufficient data for timing analysis"}

        # Find the absolute low (best buy) in last year
        min_idx = close.idxmin()
        min_price = float(close.min())
        current = float(close.iloc[-1])
        timing["best_buy_date"] = str(min_idx.date()) if hasattr(min_idx, "date") else str(min_idx)
        timing["best_buy_price"] = round(min_price, 2)
        timing["current_price"] = round(current, 2)
        timing["gain_if_bought_at_low"] = round((current - min_price) / min_price * 100, 2)

        # Find top 3 local minima (dips) in last year
        from scipy.signal import argrelextrema
        local_min_idxs = argrelextrema(close.values, np.less, order=10)[0]
        dips = []
        for i in local_min_idxs[-5:]:  # last 5 dips
            dt = close.index[i]
            price = float(close.iloc[i])
            dips.append({
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "price": round(price, 2),
            })
        timing["recent_dips"] = dips

        # Estimate next opportunity using momentum and mean-reversion
        mom_20 = float(features["momentum_20"].iloc[-1]) if "momentum_20" in features.columns else 0
        rsi = float(features.get("rsi_14", features["close"] * 0 + 50).iloc[-1])

        if rsi < 35:
            timing["next_opportunity"] = "NOW — RSI oversold, potential bounce"
            timing["opportunity_strength"] = "Strong"
        elif rsi < 45 and mom_20 < -0.05:
            timing["next_opportunity"] = "Approaching — momentum weak, watch for reversal"
            timing["opportunity_strength"] = "Moderate"
        elif mom_20 > 0.10:
            timing["next_opportunity"] = "Wait — extended rally, pullback likely in 2-6 weeks"
            timing["opportunity_strength"] = "Patient"
        else:
            timing["next_opportunity"] = "Neutral — no strong timing signal, use limit orders at support"
            timing["opportunity_strength"] = "Neutral"

        # Support level estimate (recent swing low)
        if len(dips) > 0:
            timing["support_level"] = dips[-1]["price"]
        else:
            timing["support_level"] = round(min_price * 1.02, 2)

        # Average days between dips
        if len(dips) >= 2:
            from datetime import datetime as dt_cls
            dates = [dt_cls.strptime(d["date"], "%Y-%m-%d") for d in dips]
            gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            avg_gap = sum(gaps) / len(gaps)
            timing["avg_days_between_dips"] = round(avg_gap)
            last_dip = dates[-1]
            est_next = last_dip + __import__("datetime").timedelta(days=int(avg_gap))
            timing["estimated_next_dip"] = str(est_next.date())

    except Exception as e:
        logger.warning(f"Best buy timing failed: {e}")
        timing["error"] = str(e)

    return timing


def save_analysis(result) -> str:
    """Save analysis result to history. Returns the filename."""
    _ensure_dir()
    entry = _serialize_result(result)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol = result.instrument.symbol.replace(".", "_")
    filename = f"{ts}_{symbol}.json"
    filepath = HISTORY_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved analysis to {filepath}")
    return filename


def load_history(limit: int = 50) -> list[dict[str, Any]]:
    """Load the most recent analysis runs from history."""
    _ensure_dir()
    files = sorted(HISTORY_DIR.glob("*.json"), reverse=True)[:limit]
    history = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                data["_filename"] = f.name
                history.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    return history


def load_history_for_symbol(symbol: str, limit: int = 20) -> list[dict[str, Any]]:
    """Load history for a specific symbol."""
    all_history = load_history(limit=200)
    return [h for h in all_history if h.get("symbol", "").upper() == symbol.upper()][:limit]


def clear_history() -> int:
    """Delete all history files. Returns count deleted."""
    _ensure_dir()
    files = list(HISTORY_DIR.glob("*.json"))
    for f in files:
        f.unlink()
    return len(files)
