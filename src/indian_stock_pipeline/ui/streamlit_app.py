from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from indian_stock_pipeline.core.config import get_settings
from indian_stock_pipeline.core.pipeline import StockAnalysisPipeline


def _build_price_chart(features: pd.DataFrame, exit_target: float, stop_loss: float) -> go.Figure:
    chart = go.Figure()
    chart.add_trace(
        go.Candlestick(
            x=features.index,
            open=features["open"],
            high=features["high"],
            low=features["low"],
            close=features["close"],
            name="Price",
        )
    )
    chart.add_trace(go.Scatter(x=features.index, y=features["ema_fast"], mode="lines", name="EMA 12",
                               line=dict(color="#3b82f6", width=1.5)))
    chart.add_trace(go.Scatter(x=features.index, y=features["ema_slow"], mode="lines", name="EMA 26",
                               line=dict(color="#f59e0b", width=1.5)))
    chart.add_hline(y=exit_target, line_dash="dash", line_color="#16a34a", annotation_text="Exit target")
    chart.add_hline(y=stop_loss, line_dash="dash", line_color="#dc2626", annotation_text="Stop-loss")
    chart.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10), xaxis_rangeslider_visible=False)
    return chart


def _build_equity_chart(equity_curve: pd.DataFrame) -> go.Figure:
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve["equity"], mode="lines", name="Strategy Equity",
                               line=dict(color="#8b5cf6", width=2)))
    chart.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    return chart


def _score_table(score_breakdown: dict[str, float]) -> pd.DataFrame:
    pretty_names = {
        "regime": "Regime strength",
        "trend": "Trend quality",
        "volume": "Volume confirmation",
        "momentum": "Momentum",
        "ofi": "Order-flow proxy",
        "patterns": "Pattern quality",
        "deep_model": "Deep model (PatchTST)",
        "rsi": "RSI signal",
        "adx": "ADX trend strength",
        "delivery": "Delivery % conviction",
        "pcr": "Put-Call Ratio",
        "liquidity_penalty": "Liquidity penalty",
        "anomaly_penalty": "Overextension penalty",
    }
    rows = [{"component": pretty_names.get(key, key.title()), "score": round(value, 3)} for key, value in score_breakdown.items()]
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def _render_recommendation_summary(result) -> None:
    recommendation = result.recommendation
    latest = result.features.iloc[-1]

    st.subheader("Decision Summary")
    st.info(f"**{recommendation.headline}** {recommendation.summary}")

    cols = st.columns(5)
    cols[0].metric("Action", recommendation.action)
    cols[1].metric("Confidence", f"{recommendation.confidence:.0%}")
    cols[2].metric("Setup quality", recommendation.setup_quality)
    cols[3].metric("Horizon", recommendation.horizon)
    cols[4].metric("Regime", recommendation.regime_label.replace("_", " ").title())

    levels = st.columns(5)
    levels[0].metric("Ideal entry", f"{recommendation.entry_low:.2f} - {recommendation.entry_high:.2f}")
    levels[1].metric("Stop-loss", f"{recommendation.stop_loss:.2f}")
    levels[2].metric("Trailing stop", f"{recommendation.trailing_stop:.2f}")
    levels[3].metric("Exit target", f"{recommendation.exit_target:.2f}")
    levels[4].metric("R:R ratio", f"{recommendation.risk_reward:.1f}")

    # Position sizing row
    sizing = st.columns(4)
    sizing[0].metric("Position size", f"{recommendation.position_size:.0%}")
    sizing[1].metric("Kelly fraction", f"{recommendation.kelly_fraction:.2%}")
    sizing[2].metric("VIX Z-score", f"{result.regime.vix_zscore:.2f}")
    sizing[3].metric("FII/DII flow", result.regime.fii_dii_direction.title())

    st.write(
        f"Latest close: `{latest['close']:.2f}` | ATR: `{latest['atr']:.2f}` | "
        f"1-month momentum: `{latest['momentum_20']:.2%}` | 3-month momentum: `{latest['momentum_63']:.2%}` | "
        f"RSI: `{latest.get('rsi_14', 50):.0f}` | ADX: `{latest.get('adx_14', 25):.0f}`"
    )

    if recommendation.risk_flags:
        st.warning("Main risk flags: " + " | ".join(recommendation.risk_flags))


def _render_signal_explanations(result) -> None:
    recommendation = result.recommendation
    st.subheader("Why The Model Said This")
    for note in recommendation.notes:
        st.write(f"- {note}")

    st.dataframe(_score_table(recommendation.score_breakdown), use_container_width=True, hide_index=True)

    with st.expander("What these checks mean"):
        for key, explanation in recommendation.feature_explanations.items():
            st.write(f"**{key}**: {explanation}")


def _render_backtest(result) -> None:
    st.subheader("Strategy Backtest")
    backtest = result.backtest
    if not backtest.enabled:
        st.write(backtest.notes or "Backtest unavailable.")
        return

    cols = st.columns(6)
    cols[0].metric("Strategy return", f"{(backtest.total_return or 0.0):.2%}")
    cols[1].metric("Benchmark return", f"{(backtest.benchmark_return or 0.0):.2%}")
    cols[2].metric("Annualized", f"{(backtest.annualized_return or 0.0):.2%}")
    cols[3].metric("Sharpe", f"{(backtest.sharpe_ratio or 0.0):.2f}")
    cols[4].metric("Max drawdown", f"{(backtest.max_drawdown or 0.0):.2%}")
    cols[5].metric("Win rate", f"{(backtest.win_rate or 0.0):.2%}")

    meta = st.columns(5)
    meta[0].metric("Trades", f"{backtest.total_trades or 0}")
    meta[1].metric("Avg trade", f"{(backtest.avg_trade_return or 0.0):.2%}")
    meta[2].metric("Expectancy", f"{(backtest.expectancy or 0.0):.2%}")
    meta[3].metric("Last signal", backtest.last_signal or "N/A")
    # Monte Carlo significance
    mc_color = "normal" if backtest.monte_carlo_p_value is None else ("off" if backtest.monte_carlo_p_value > 0.1 else "normal")
    meta[4].metric("MC p-value", f"{(backtest.monte_carlo_p_value or 1.0):.3f}")

    if backtest.monte_carlo_p_value is not None:
        if backtest.monte_carlo_p_value < 0.05:
            st.success(f"Monte Carlo test: statistically significant (p={backtest.monte_carlo_p_value:.3f}). Strategy edge is unlikely due to chance.")
        elif backtest.monte_carlo_p_value < 0.10:
            st.info(f"Monte Carlo test: marginally significant (p={backtest.monte_carlo_p_value:.3f}). Some evidence of real edge.")
        else:
            st.warning(f"Monte Carlo test: not significant (p={backtest.monte_carlo_p_value:.3f}). Strategy returns could be explained by random chance.")

    # Walk-forward fold metrics
    if backtest.fold_metrics:
        st.write("**Walk-Forward Fold Metrics**")
        fold_rows = [
            {
                "Fold": fm.fold_index + 1,
                "Train": f"{fm.train_start} → {fm.train_end}",
                "Test": f"{fm.test_start} → {fm.test_end}",
                "Return": f"{fm.total_return:.2%}",
                "Sharpe": f"{fm.sharpe_ratio:.2f}",
                "Max DD": f"{fm.max_drawdown:.2%}",
                "Win Rate": f"{fm.win_rate:.0%}",
                "Trades": fm.total_trades,
            }
            for fm in backtest.fold_metrics
        ]
        st.dataframe(pd.DataFrame(fold_rows), use_container_width=True, hide_index=True)

    if not backtest.equity_curve.empty:
        st.plotly_chart(_build_equity_chart(backtest.equity_curve), use_container_width=True)

    if backtest.trade_log:
        st.write("Recent trades")
        st.dataframe(pd.DataFrame(backtest.trade_log), use_container_width=True, hide_index=True)

    if backtest.notes:
        st.caption(backtest.notes)


def _render_market_screener(result) -> None:
    st.subheader("Top Market Ideas")
    if result.screener is None:
        st.write("Market screener was skipped for this run. Enable it before analysis to scan for top ideas.")
        return

    screener = result.screener
    st.write(f"Universe: `{screener.universe_name}` | Scanned: `{screener.scanned_count}` stocks")

    long_col, short_col = st.columns(2)
    with long_col:
        st.write("Top 5 long-term buy candidates")
        if screener.long_candidates:
            long_rows = [
                {
                    "symbol": idea.symbol,
                    "action": idea.action,
                    "confidence": f"{idea.confidence:.0%}",
                    "setup": idea.setup_quality,
                    "close": round(idea.latest_close, 2),
                    "1M": f"{idea.momentum_20:.2%}",
                    "3M": f"{idea.momentum_63:.2%}",
                    "summary": idea.summary,
                }
                for idea in screener.long_candidates
            ]
            st.dataframe(pd.DataFrame(long_rows), use_container_width=True, hide_index=True)
        else:
            st.write("No strong long ideas were found in the scanned universe.")

    with short_col:
        st.write("Top 5 short-term sell / avoid candidates")
        if screener.short_candidates:
            short_rows = [
                {
                    "symbol": idea.symbol,
                    "action": idea.action,
                    "confidence": f"{idea.confidence:.0%}",
                    "setup": idea.setup_quality,
                    "close": round(idea.latest_close, 2),
                    "1M": f"{idea.momentum_20:.2%}",
                    "3M": f"{idea.momentum_63:.2%}",
                    "summary": idea.summary,
                }
                for idea in screener.short_candidates
            ]
            st.dataframe(pd.DataFrame(short_rows), use_container_width=True, hide_index=True)
        else:
            st.write("No clear weak short-term ideas were found in the scanned universe.")

    if screener.notes:
        for note in screener.notes:
            st.caption(note)


def _render_advanced(result) -> None:
    latest = result.features.iloc[-1]
    st.subheader("Advanced Diagnostics")

    left, right = st.columns([1.2, 1.0])
    with left:
        st.write(
            f"Resolved from `{result.instrument.source}` on `{result.instrument.exchange}`. "
            f"Daily source: `{result.market_data.metadata.get('daily_source', 'n/a')}` | "
            f"Intraday source: `{result.market_data.metadata.get('intraday_source', 'n/a')}`"
        )
        st.write(
            f"Regime state: `{result.regime.current_state}` | Trend state: `{result.regime.trend_state}` | "
            f"Trending probability: `{result.regime.trending_probability:.0%}` | "
            f"Label: **{result.regime.regime_label.replace('_', ' ').title()}**"
        )
        st.write(
            f"Kalman filtered price: `{result.regime.kalman_price:.2f}` | "
            f"Kalman target: `{result.regime.kalman_target:.2f}` | "
            f"VIX Z-score: `{result.regime.vix_zscore:.2f}` | "
            f"FII/DII: `{result.regime.fii_dii_direction}`"
        )
        st.write(
            f"Pattern score: `{result.patterns.matrix_profile_score:.2f}` | "
            f"DTW similarity: `{result.patterns.dtw_similarity:.2f}` | "
            f"Anomaly: `{result.patterns.anomaly_score:.2f}`"
        )
        if result.patterns.candlestick_patterns:
            st.write(f"Candlestick patterns: **{', '.join(result.patterns.candlestick_patterns)}**")
        if result.patterns.multi_window_mp_scores:
            st.write(f"Multi-window MP scores: {', '.join(f'w={k}: {v:.2f}' for k, v in result.patterns.multi_window_mp_scores.items())}")

        st.write("---")
        st.write("**New SOTA Features**")
        sota_features = {
            "RSI (14)": f"{latest.get('rsi_14', 'N/A'):.1f}" if isinstance(latest.get("rsi_14"), float) else "N/A",
            "ADX (14)": f"{latest.get('adx_14', 'N/A'):.1f}" if isinstance(latest.get("adx_14"), float) else "N/A",
            "Choppiness": f"{latest.get('choppiness_14', 'N/A'):.1f}" if isinstance(latest.get("choppiness_14"), float) else "N/A",
            "GK Volatility": f"{latest.get('garman_klass_vol', 'N/A'):.2%}" if isinstance(latest.get("garman_klass_vol"), float) else "N/A",
            "Parkinson Vol": f"{latest.get('parkinson_vol', 'N/A'):.2%}" if isinstance(latest.get("parkinson_vol"), float) else "N/A",
            "MACD Histogram": f"{latest.get('macd_histogram', 'N/A'):.2f}" if isinstance(latest.get("macd_histogram"), float) else "N/A",
            "Delivery %": f"{latest.get('delivery_pct', 'N/A'):.0%}" if isinstance(latest.get("delivery_pct"), float) else "N/A",
            "Put-Call Ratio": f"{latest.get('put_call_ratio', 'N/A'):.2f}" if isinstance(latest.get("put_call_ratio"), float) else "N/A",
        }
        st.dataframe(pd.DataFrame(list(sota_features.items()), columns=["Feature", "Value"]),
                      use_container_width=True, hide_index=True)

        if result.forecast.enabled:
            st.write(f"**PatchTST Forecast**: predicted return `{result.forecast.predicted_return:.2%}` | "
                     f"confidence `{result.forecast.confidence:.2%}` | device `{result.forecast.device}` | "
                     f"training loss `{result.forecast.training_loss:.4f}`")

    with right:
        st.write("NSE context snapshots")
        st.json(
            {
                "live_quote": result.alternative_data.live_quote or {"status": "unavailable"},
                "option_chain": result.alternative_data.option_chain or {"status": "unavailable"},
                "fii_dii": result.alternative_data.fii_dii or [{"status": "unavailable"}],
                "bhavcopy_delivery": result.alternative_data.bhavcopy_delivery or {"status": "unavailable"},
                "bhavcopy_fo": result.alternative_data.bhavcopy_fo or {"status": "unavailable"},
            }
        )

    with st.expander("Latest feature row"):
        st.dataframe(latest.rename("value").to_frame(), use_container_width=True)


def render() -> None:
    settings = get_settings()
    pipeline = StockAnalysisPipeline(settings)

    st.set_page_config(page_title="Indian Stock Analysis Pipeline — SOTA Edition", layout="wide")
    st.title("🚀 Indian Stock Analysis Pipeline — SOTA Edition")
    st.caption(
        "PatchTST deep model · Regime-adaptive ensemble · Fractional Kelly sizing · "
        "Walk-forward purged CV · Monte Carlo significance · India VIX / FII-DII / Delivery% / PCR integration"
    )

    with st.sidebar:
        st.subheader("Runtime")
        st.write(f"Data provider mode: `{settings.data_provider}`")
        st.write(f"Default exchange: `{settings.default_exchange}`")
        st.write(f"History period: `{settings.history_period}`")
        st.write("Primary stack: `jugaad-data + nsepython`")
        st.write(f"Deep model: `{'PatchTST (GPU)' if settings.enable_torch_model else 'Disabled'}`")
        st.write(f"Walk-forward folds: `{settings.wf_folds}`")
        st.write(f"Kelly cap: `{settings.kelly_fraction_cap:.0%}`")
        st.info("No paid broker key is required. Free open-source data only.")

        st.subheader("SOTA Upgrades")
        st.markdown("""
        - ✅ PatchTST Transformer (FP16)
        - ✅ 30+ quant features
        - ✅ RSI, ADX, Choppiness Index
        - ✅ Garman-Klass & Parkinson vol
        - ✅ Delivery %, PCR, India VIX
        - ✅ FII/DII flow integration
        - ✅ Multi-signal HMM regime detection
        - ✅ Regime persistence filter
        - ✅ 2-state Kalman (price + velocity)
        - ✅ FastDTW with Sakoe-Chiba band
        - ✅ Multi-window matrix profile
        - ✅ Candlestick pattern recognition
        - ✅ Regime-adaptive signal weights
        - ✅ Fractional Kelly position sizing
        - ✅ VIX-adaptive thresholds
        - ✅ Walk-forward purged CV
        - ✅ Monte Carlo significance test
        """)

    with st.form("analysis-form"):
        company_query = st.text_input("Company name or trading symbol", placeholder="Bharti Airtel or BHARTIARTL.NS")
        include_screener = st.checkbox("Also scan the market for top 5 buy/sell ideas", value=True)
        submitted = st.form_submit_button("Analyze")

    if not submitted:
        st.markdown(
            """
            ### What this app does
            - Explains the decision in plain English before showing raw quant internals
            - Uses **PatchTST Transformer** for directional forecasting (GPU-accelerated)
            - **Regime-adaptive ensemble** adjusts signal weights for bull/bear/sideways/volatile conditions
            - **Fractional Kelly** position sizing with VIX-based regime caps
            - Walk-forward **purged cross-validation** backtesting with **Monte Carlo** significance
            - Integrates **India VIX, FII/DII flows, delivery %, put-call ratio** from free open-source feeds
            - Backtests the same strategy family on rolling historical data
            - Shows top 5 long-term buy and top 5 short-term weak candidates
            """
        )
        return

    try:
        with st.spinner("Running analysis, backtest, and market scan..."):
            result = pipeline.analyze(company_query, include_screener=include_screener)
    except Exception as exc:  # pragma: no cover - Streamlit UX path
        st.error(str(exc))
        return

    st.plotly_chart(
        _build_price_chart(result.features.tail(180), result.recommendation.exit_target, result.recommendation.stop_loss),
        use_container_width=True,
    )

    tab_summary, tab_backtest, tab_market, tab_advanced = st.tabs(
        ["Summary", "Backtest", "Market Ideas", "Advanced"]
    )

    with tab_summary:
        _render_recommendation_summary(result)
        _render_signal_explanations(result)

    with tab_backtest:
        _render_backtest(result)

    with tab_market:
        _render_market_screener(result)

    with tab_advanced:
        _render_advanced(result)


render()
