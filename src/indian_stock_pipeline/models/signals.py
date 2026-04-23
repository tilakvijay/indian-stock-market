from __future__ import annotations

import numpy as np
import pandas as pd

from indian_stock_pipeline.core.schemas import ForecastSummary, PatternSummary, Recommendation, RegimeSummary


def _bounded(value: float | pd.Series, lower: float = 0.0, upper: float = 1.0):
    return np.clip(value, lower, upper)


def _score_from_return(value: float | pd.Series, scale: float = 0.03):
    return _bounded((value + scale) / (2 * scale))


# ---------------------------------------------------------------------------
# Regime-conditional weight sets — SOTA adaptive ensemble
# ---------------------------------------------------------------------------
_REGIME_WEIGHTS = {
    "bull_low_vol": {
        "regime": 0.18, "trend": 0.22, "volume": 0.10, "momentum": 0.18,
        "ofi": 0.08, "patterns": 0.10, "deep_model": 0.14,
        "rsi": 0.05, "adx": 0.05, "delivery": 0.04, "pcr": 0.03,
        "liq_penalty": 0.03, "anom_penalty": 0.03,
    },
    "bull_high_vol": {
        "regime": 0.22, "trend": 0.15, "volume": 0.12, "momentum": 0.10,
        "ofi": 0.10, "patterns": 0.08, "deep_model": 0.08,
        "rsi": 0.06, "adx": 0.04, "delivery": 0.06, "pcr": 0.06,
        "liq_penalty": 0.05, "anom_penalty": 0.05,
    },
    "bear_low_vol": {
        "regime": 0.25, "trend": 0.12, "volume": 0.14, "momentum": 0.08,
        "ofi": 0.12, "patterns": 0.08, "deep_model": 0.06,
        "rsi": 0.06, "adx": 0.04, "delivery": 0.05, "pcr": 0.06,
        "liq_penalty": 0.05, "anom_penalty": 0.06,
    },
    "bear_high_vol": {
        "regime": 0.28, "trend": 0.08, "volume": 0.12, "momentum": 0.06,
        "ofi": 0.10, "patterns": 0.06, "deep_model": 0.04,
        "rsi": 0.06, "adx": 0.04, "delivery": 0.06, "pcr": 0.08,
        "liq_penalty": 0.07, "anom_penalty": 0.08,
    },
    "sideways": {
        "regime": 0.15, "trend": 0.10, "volume": 0.12, "momentum": 0.08,
        "ofi": 0.14, "patterns": 0.12, "deep_model": 0.10,
        "rsi": 0.08, "adx": 0.04, "delivery": 0.05, "pcr": 0.05,
        "liq_penalty": 0.05, "anom_penalty": 0.05,
    },
}
# Default fallback weights
_DEFAULT_WEIGHTS = _REGIME_WEIGHTS["sideways"]


def _get_weights(regime_label: str) -> dict[str, float]:
    return _REGIME_WEIGHTS.get(regime_label, _DEFAULT_WEIGHTS)


# ---------------------------------------------------------------------------
# Fractional Kelly position sizing
# ---------------------------------------------------------------------------

def _compute_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                            kelly_cap: float = 0.25) -> float:
    """Compute fractional Kelly criterion for position sizing.

    Kelly formula: f* = (b*p - q) / b
    where p=win_rate, q=1-p, b=avg_win/avg_loss
    Capped at kelly_cap (default 25%) to reduce volatility.
    """
    if avg_loss == 0 or win_rate <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss
    p = win_rate
    q = 1.0 - p
    kelly = (b * p - q) / b
    return float(np.clip(kelly * kelly_cap, 0.0, kelly_cap))


def _estimate_trade_stats(features: pd.DataFrame) -> tuple[float, float, float]:
    """Estimate win rate, avg win, avg loss from recent returns for Kelly sizing."""
    returns = features["simple_return"].dropna().tail(252)
    if len(returns) < 20:
        return 0.5, 0.01, 0.01
    wins = returns[returns > 0]
    losses = returns[returns < 0].abs()
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.5
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.01
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.01
    return win_rate, avg_win, avg_loss


# ---------------------------------------------------------------------------
# Regime-based position size cap
# ---------------------------------------------------------------------------

def _regime_position_cap(regime_label: str, vix_zscore: float) -> float:
    """Cap position size based on regime and VIX."""
    base_caps = {
        "bull_low_vol": 1.0,
        "bull_high_vol": 0.6,
        "bear_low_vol": 0.4,
        "bear_high_vol": 0.2,
        "sideways": 0.5,
        "unknown": 0.3,
    }
    cap = base_caps.get(regime_label, 0.3)
    # Further reduce in extreme VIX
    if vix_zscore > 2.0:
        cap *= 0.5
    elif vix_zscore > 1.0:
        cap *= 0.75
    return float(np.clip(cap, 0.1, 1.0))


def build_signal_frame(features: pd.DataFrame, regime_label: str = "unknown") -> pd.DataFrame:
    weights = _get_weights(regime_label)

    signal_frame = pd.DataFrame(index=features.index)

    signal_frame["trend"] = _bounded((features["trend_strength"] + 0.04) / 0.08)
    signal_frame["volume"] = _bounded(features["volume_confirmation"] / 2.0)
    signal_frame["momentum"] = (
        0.55 * _score_from_return(features["momentum_20"], scale=0.12)
        + 0.45 * _score_from_return(features["momentum_63"], scale=0.25)
    )
    signal_frame["ofi"] = _bounded((features["ofi_proxy"] + 2.0) / 4.0)
    signal_frame["regime"] = _bounded(
        0.45
        + (7.5 * features["trend_strength"])
        + (1.1 * (features["hurst"] - 0.5))
        - (0.3 * features["downside_vol_20"].fillna(0.0))
        + (0.12 * features["trend_persistence_20"].fillna(0.5))
    )
    signal_frame["patterns"] = (
        0.35 * _bounded((features["breakout_55"] + 0.05) / 0.10)
        + 0.30 * _bounded((features["vwap_deviation"] + 0.03) / 0.06)
        + 0.35 * _bounded((features["trend_persistence_20"] - 0.35) / 0.45)
    )
    signal_frame["deep_model"] = 0.0
    signal_frame["liquidity_penalty"] = _bounded(features["amihud_illiquidity"] * 1_000_000)
    signal_frame["anomaly_penalty"] = _bounded(features["close_zscore_20"].abs() / 3.0)

    # --- NEW SOTA signal components ---

    # RSI component — overbought/oversold as score
    if "rsi_14" in features.columns:
        signal_frame["rsi"] = _bounded(features["rsi_14"] / 100.0)
    else:
        signal_frame["rsi"] = 0.5

    # ADX component — trend strength
    if "adx_14" in features.columns:
        signal_frame["adx"] = _bounded(features["adx_14"] / 60.0)
    else:
        signal_frame["adx"] = 0.5

    # Delivery percentage — institutional conviction
    if "delivery_pct" in features.columns:
        signal_frame["delivery"] = _bounded(features["delivery_pct"])
    else:
        signal_frame["delivery"] = 0.5

    # Put-Call Ratio — PCR > 1.2 = bullish, PCR < 0.7 = bearish
    if "put_call_ratio" in features.columns:
        signal_frame["pcr"] = _bounded((features["put_call_ratio"] - 0.5) / 1.0)
    else:
        signal_frame["pcr"] = 0.5

    # --- Regime-adaptive composite score ---
    w = weights
    signal_frame["score"] = (
        (w["regime"] * signal_frame["regime"])
        + (w["trend"] * signal_frame["trend"])
        + (w["volume"] * signal_frame["volume"])
        + (w["momentum"] * signal_frame["momentum"])
        + (w["ofi"] * signal_frame["ofi"])
        + (w["patterns"] * signal_frame["patterns"])
        + (w["deep_model"] * signal_frame["deep_model"])
        + (w["rsi"] * signal_frame["rsi"])
        + (w["adx"] * signal_frame["adx"])
        + (w["delivery"] * signal_frame["delivery"])
        + (w["pcr"] * signal_frame["pcr"])
        - (w["liq_penalty"] * signal_frame["liquidity_penalty"])
        - (w["anom_penalty"] * signal_frame["anomaly_penalty"])
    )
    signal_frame["score"] = _bounded(signal_frame["score"])

    # --- Dynamic entry/exit thresholds based on VIX ---
    vix_z = features.get("india_vix_zscore", pd.Series(0.0, index=features.index))
    # Higher VIX = more conservative thresholds
    entry_threshold = _bounded(0.64 + 0.04 * vix_z, 0.58, 0.78)
    watch_threshold = _bounded(0.55 + 0.03 * vix_z, 0.50, 0.68)
    exit_threshold = _bounded(0.48 + 0.02 * vix_z, 0.42, 0.58)

    signal_frame["long_entry"] = (
        (signal_frame["score"] >= entry_threshold)
        & (signal_frame["regime"] >= 0.52)
        & (features["ofi_proxy"] > 0)
        & (features["close"] >= features["ema_fast"])
    )
    signal_frame["watch"] = signal_frame["score"] >= watch_threshold
    signal_frame["short_bias"] = (
        (signal_frame["score"] <= 0.38)
        & (features["trend_strength"] < 0)
        & (features["close"] < features["ema_slow"])
    )
    signal_frame["exit"] = (
        (signal_frame["score"] <= exit_threshold)
        | (features["close"] < features["ema_slow"])
        | (features["close"] < (features["close"].shift(1) - 1.5 * features["atr"]))
    )
    return signal_frame.fillna(0.0)


def build_recommendation(
    features: pd.DataFrame,
    regime: RegimeSummary,
    patterns: PatternSummary,
    forecast: ForecastSummary,
    kelly_cap: float = 0.25,
) -> Recommendation:
    latest = features.iloc[-1]
    signal_frame = build_signal_frame(features, regime_label=regime.regime_label)
    latest_signal = signal_frame.iloc[-1].copy()

    latest_signal["regime"] = float(_bounded((latest_signal["regime"] + regime.trending_probability) / 2.0))
    latest_signal["patterns"] = float(
        _bounded((latest_signal["patterns"] + patterns.matrix_profile_score + patterns.dtw_similarity) / 3.0)
    )
    latest_signal["deep_model"] = float(
        _bounded(forecast.confidence * _score_from_return(forecast.predicted_return, scale=0.02))
    )

    # Boost from delivery percentage and PCR
    if float(latest.get("delivery_pct", 0.5)) > 0.6:
        latest_signal["delivery"] = float(_bounded(latest_signal.get("delivery", 0.5) * 1.15))
    pcr = float(latest.get("put_call_ratio", 1.0))
    if pcr > 1.2:
        latest_signal["pcr"] = float(_bounded(latest_signal.get("pcr", 0.5) * 1.1))
    elif pcr < 0.7:
        latest_signal["pcr"] = float(_bounded(latest_signal.get("pcr", 0.5) * 0.85))

    score_breakdown = {
        "regime": float(latest_signal["regime"]),
        "trend": float(latest_signal["trend"]),
        "volume": float(latest_signal["volume"]),
        "momentum": float(latest_signal["momentum"]),
        "ofi": float(latest_signal["ofi"]),
        "patterns": float(latest_signal["patterns"]),
        "deep_model": float(latest_signal["deep_model"]),
        "rsi": float(latest_signal.get("rsi", 0.5)),
        "adx": float(latest_signal.get("adx", 0.5)),
        "delivery": float(latest_signal.get("delivery", 0.5)),
        "pcr": float(latest_signal.get("pcr", 0.5)),
        "liquidity_penalty": float(latest_signal["liquidity_penalty"]),
        "anomaly_penalty": float(max(latest_signal["anomaly_penalty"], patterns.anomaly_score)),
    }

    w = _get_weights(regime.regime_label)
    confidence = float(
        _bounded(
            (w["regime"] * score_breakdown["regime"])
            + (w["trend"] * score_breakdown["trend"])
            + (w["volume"] * score_breakdown["volume"])
            + (w["momentum"] * score_breakdown["momentum"])
            + (w["ofi"] * score_breakdown["ofi"])
            + (w["patterns"] * score_breakdown["patterns"])
            + (w["deep_model"] * score_breakdown["deep_model"])
            + (w["rsi"] * score_breakdown["rsi"])
            + (w["adx"] * score_breakdown["adx"])
            + (w["delivery"] * score_breakdown["delivery"])
            + (w["pcr"] * score_breakdown["pcr"])
            - (w["liq_penalty"] * score_breakdown["liquidity_penalty"])
            - (w["anom_penalty"] * score_breakdown["anomaly_penalty"])
        )
    )

    close = float(latest["close"])
    atr = float(latest["atr"])
    atr = atr if np.isfinite(atr) and atr > 0 else close * 0.03

    # --- Dynamic thresholds based on VIX ---
    vix_z = regime.vix_zscore
    buy_thresh = float(np.clip(0.68 + 0.04 * vix_z, 0.62, 0.80))
    watch_thresh = float(np.clip(0.55 + 0.03 * vix_z, 0.50, 0.68))
    reduce_thresh = float(np.clip(0.42 + 0.02 * vix_z, 0.36, 0.50))

    action = "HOLD"
    setup_quality = "Average"
    if confidence >= buy_thresh and close >= regime.kalman_price and float(latest["ofi_proxy"]) > 0:
        action = "BUY"
        setup_quality = "High"
    elif confidence >= watch_thresh:
        action = "WATCH"
        setup_quality = "Developing"
    elif confidence < reduce_thresh:
        action = "REDUCE"
        setup_quality = "Weak"

    entry_low = close - (0.35 * atr)
    entry_high = close + (0.35 * atr)
    stop_loss = close - (2.0 * atr)
    trailing_stop = close - (1.35 * atr)
    exit_target = max(regime.kalman_target, close + (2.4 * atr))
    risk_reward = max((exit_target - close) / max(close - stop_loss, 1e-6), 0.0)

    # --- Fractional Kelly position sizing ---
    win_rate, avg_win, avg_loss = _estimate_trade_stats(features)
    kelly_fraction = _compute_kelly_fraction(win_rate, avg_win, avg_loss, kelly_cap)
    regime_cap = _regime_position_cap(regime.regime_label, regime.vix_zscore)
    position_size = float(np.clip(kelly_fraction * (1.0 / kelly_cap), 0.0, 1.0) * regime_cap)

    notes: list[str] = []
    risk_flags: list[str] = []

    if regime.current_state == regime.trend_state:
        notes.append("The regime filter says the stock is trading in its stronger trend state.")
    else:
        notes.append("The regime filter is not aligned, so the system is intentionally cautious.")
        risk_flags.append("Regime alignment is weak.")

    notes.append(f"Market regime: **{regime.regime_label.replace('_', ' ').title()}**.")

    if regime.fii_dii_direction == "buying":
        notes.append("FII/DII flow is net buying — institutional support is present.")
    elif regime.fii_dii_direction == "selling":
        notes.append("FII/DII flow is net selling — institutional headwind.")
        risk_flags.append("Institutional flows are negative.")

    if regime.vix_zscore > 1.0:
        notes.append(f"India VIX is elevated (Z-score: {regime.vix_zscore:.1f}), thresholds have been tightened.")
        risk_flags.append("Elevated market volatility (VIX).")

    if latest["hurst"] > 0.55:
        notes.append("Trend persistence is positive, which supports swing continuation and long-term holding.")
    elif latest["hurst"] < 0.45:
        notes.append("This stock behaves more like a mean-reverter right now, so momentum trades are less reliable.")
        risk_flags.append("Behavior is more mean-reverting than trending.")
    else:
        notes.append("The trend structure is mixed, so follow-through may be inconsistent.")

    if float(latest["volume_confirmation"]) > 1.1:
        notes.append("Volume is healthy versus its recent average, so price moves are better confirmed.")
    else:
        notes.append("Volume is below ideal confirmation, which reduces conviction.")
        risk_flags.append("Volume confirmation is soft.")

    if float(latest.get("delivery_pct", 0.5)) > 0.6:
        notes.append(f"Delivery percentage is high ({float(latest['delivery_pct']):.0%}), indicating institutional conviction.")
    elif float(latest.get("delivery_pct", 0.5)) < 0.3:
        risk_flags.append("Low delivery percentage — mostly speculative trading.")

    if patterns.candlestick_patterns:
        notes.append(f"Candlestick patterns detected: {', '.join(patterns.candlestick_patterns)}.")

    if float(latest["drawdown_252"]) < -0.20:
        risk_flags.append("The stock is still materially below its 1-year high.")

    if float(latest["downside_vol_20"]) > float(latest["realized_vol"]) * 0.85:
        risk_flags.append("Recent downside volatility is elevated.")

    rsi = float(latest.get("rsi_14", 50))
    if rsi > 75:
        risk_flags.append(f"RSI is overbought ({rsi:.0f}).")
    elif rsi < 25:
        notes.append(f"RSI is oversold ({rsi:.0f}), potential for mean-reversion bounce.")

    if forecast.enabled:
        notes.append(f"PatchTST deep model adds a directional nudge of {forecast.predicted_return:.2%} (loss: {forecast.training_loss:.4f}).")
    else:
        notes.append("The current signal is driven by classical quant features rather than the optional deep model.")

    notes.append(f"Position sizing: **{position_size:.0%}** of capital (Kelly fraction: {kelly_fraction:.2%}, regime cap: {regime_cap:.0%}).")

    if action == "BUY":
        headline = "Constructive setup with a buy bias."
        summary = "The model sees enough trend, momentum, and structure support to justify accumulation with defined risk."
        horizon = "Long bias with swing-to-position horizon"
    elif action == "WATCH":
        headline = "Interesting, but not fully confirmed yet."
        summary = "The stock is close to a tradable setup, but the system wants better alignment before upgrading it to a buy."
        horizon = "Watchlist candidate"
    elif action == "REDUCE":
        headline = "Weak tactical setup right now."
        summary = "The model sees enough weakness or misalignment that fresh buying is not attractive, and existing exposure should be handled carefully."
        horizon = "Short-term defensive stance"
    else:
        headline = "Neutral setup."
        summary = "The stock does not currently offer a strong edge in either direction."
        horizon = "Wait for a clearer setup"

    feature_explanations = {
        "Hurst exponent": "Above 0.55 usually means the trend has persistence; below 0.45 often means price snaps back instead of following through.",
        "Trend strength": "Compares fast and slow trend filters. Positive values mean upside trend pressure is stronger.",
        "Momentum": "Measures whether the stock has actually been moving up over the last 1 to 3 months.",
        "OFI proxy": "A volume-and-candle proxy for whether buyers or sellers were more aggressive recently.",
        "ATR": "Average True Range is the typical daily move size and is used to place stops and targets.",
        "Matrix profile / DTW": "These try to compare the current price shape with past patterns to see whether the move looks constructive or unstable.",
        "RSI": "Relative Strength Index — measures overbought (>70) or oversold (<30) conditions.",
        "ADX": "Average Directional Index — ADX above 25 indicates a strong trend regardless of direction.",
        "Choppiness Index": "Below 38.2 = strong trend, above 61.8 = choppy/range-bound market.",
        "Delivery %": "India-specific: high delivery % (>60%) means more shares were actually delivered (institutional conviction).",
        "Put-Call Ratio": "PCR > 1.2 often means put writers are confident (bullish), PCR < 0.7 suggests bearish sentiment.",
        "Kelly fraction": "Mathematical optimal position size based on win rate and payoff ratio, capped to reduce drawdowns.",
        "India VIX": "VIX Z-score above 1 means fear is elevated; the system tightens entry thresholds automatically.",
    }

    return Recommendation(
        action=action,
        confidence=confidence,
        entry_low=float(entry_low),
        entry_high=float(entry_high),
        stop_loss=float(stop_loss),
        trailing_stop=float(trailing_stop),
        exit_target=float(exit_target),
        risk_reward=float(risk_reward),
        notes=notes,
        score_breakdown=score_breakdown,
        headline=headline,
        summary=summary,
        horizon=horizon,
        risk_flags=risk_flags,
        setup_quality=setup_quality,
        feature_explanations=feature_explanations,
        position_size=position_size,
        kelly_fraction=kelly_fraction,
        regime_label=regime.regime_label,
    )
