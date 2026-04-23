from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    from hurst import compute_Hc
except ImportError:  # pragma: no cover - dependency guarded for runtime flexibility
    compute_Hc = None


def _rolling_autocorr(series: pd.Series, window: int, lag: int) -> pd.Series:
    return series.rolling(window).apply(lambda values: pd.Series(values).autocorr(lag=lag), raw=False)


def _rolling_hurst(series: pd.Series, window: int = 100) -> pd.Series:
    def _compute(values: np.ndarray) -> float:
        clean = pd.Series(values).dropna()
        if len(clean) < max(window // 2, 20):
            return np.nan
        if compute_Hc is None:
            lags = np.arange(2, min(20, len(clean) // 2))
            if len(lags) < 5:
                return np.nan
            tau = [np.std(clean.diff(lag).dropna()) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return float(poly[0] * 2.0)
        hurst_value, _, _ = compute_Hc(clean.values, kind="price", simplified=True)
        return float(hurst_value)

    return series.rolling(window).apply(_compute, raw=True)


def _realized_volatility(daily: pd.DataFrame, intraday: pd.DataFrame) -> pd.Series:
    if intraday.empty:
        return daily["close"].pct_change().rolling(20).std() * math.sqrt(252)

    intraday = intraday.copy()
    intraday["ret"] = np.log(intraday["close"]).diff()
    daily_rv = intraday["ret"].groupby(intraday.index.normalize()).apply(lambda values: np.sqrt(np.nansum(values**2)))
    aligned = daily_rv.reindex(daily.index.normalize())
    aligned.index = daily.index
    return aligned.ffill().fillna(daily["close"].pct_change().rolling(20).std() * math.sqrt(252))


def _roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
    delta = close.diff()

    def _estimate(values: np.ndarray) -> float:
        series = pd.Series(values)
        covariance = np.cov(series[1:], series[:-1])[0, 1]
        if covariance >= 0:
            return 0.0
        return float(2.0 * np.sqrt(-covariance))

    return delta.rolling(window).apply(_estimate, raw=True)


# ---------------------------------------------------------------------------
# NEW SOTA FEATURES
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — classic momentum oscillator."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength regardless of direction."""
    plus_dm = high.diff().clip(lower=0.0)
    minus_dm = (-low.diff()).clip(lower=0.0)
    # Zero out whichever is smaller
    plus_dm[plus_dm < minus_dm] = 0.0
    minus_dm[minus_dm < plus_dm] = 0.0

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_smooth = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr_smooth.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_values = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return adx_values


def _choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Choppiness Index — distinguishes trending vs range-bound (high=choppy, low=trending)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    hi_range = high.rolling(period).max() - low.rolling(period).min()
    chop = 100.0 * np.log10(atr_sum / hi_range.replace(0, np.nan)) / np.log10(period)
    return chop


def _garman_klass_vol(high: pd.Series, low: pd.Series, close: pd.Series,
                      open_price: pd.Series, window: int = 20) -> pd.Series:
    """Garman-Klass volatility — more efficient than close-to-close."""
    log_hl = (np.log(high / low.replace(0, np.nan))) ** 2
    log_co = (np.log(close / open_price.replace(0, np.nan))) ** 2
    gk = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    return np.sqrt(gk.rolling(window).mean() * 252)


def _parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson volatility — uses high-low range, better for illiquid stocks."""
    log_hl_sq = (np.log(high / low.replace(0, np.nan))) ** 2
    return np.sqrt(log_hl_sq.rolling(window).mean() / (4.0 * np.log(2.0)) * 252)


def _macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD Histogram — rate of change of MACD, captures momentum acceleration."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def compute_features(daily: pd.DataFrame, intraday: pd.DataFrame,
                     alternative_data: dict | None = None) -> pd.DataFrame:
    """Compute all features including 10 new SOTA features.

    Parameters
    ----------
    daily : pd.DataFrame
        OHLCV daily bars.
    intraday : pd.DataFrame
        OHLCV intraday bars (can be empty).
    alternative_data : dict or None
        Dictionary with keys like ``delivery_pct``, ``put_call_ratio``,
        ``india_vix``, ``fii_dii_net``.  All values fall back to neutral
        when unavailable.
    """
    if daily.empty:
        raise ValueError("No daily market data was returned for the selected instrument.")

    alt = alternative_data or {}
    features = daily.copy()

    # --- Original features ---
    features["return"] = np.log(features["close"]).diff()
    features["simple_return"] = features["close"].pct_change()
    features["realized_vol"] = _realized_volatility(features, intraday)
    features["hurst"] = _rolling_hurst(features["close"], window=100)
    features["autocorr_lag_1"] = _rolling_autocorr(features["return"], window=60, lag=1)
    features["autocorr_lag_5"] = _rolling_autocorr(features["return"], window=60, lag=5)
    features["autocorr_lag_10"] = _rolling_autocorr(features["return"], window=80, lag=10)

    features["ofi_proxy"] = np.sign(features["close"] - features["open"]) * (
        features["volume"] / features["volume"].rolling(20).mean().replace(0, np.nan)
    )
    features["amihud_illiquidity"] = (
        features["simple_return"].abs() / (features["close"] * features["volume"].replace(0, np.nan))
    ).rolling(20).mean()
    features["roll_spread"] = _roll_spread(features["close"], window=20)

    typical_price = (features["high"] + features["low"] + features["close"]) / 3.0
    rolling_vwap = (typical_price * features["volume"]).rolling(20).sum() / features["volume"].rolling(20).sum()
    features["vwap_deviation"] = (features["close"] - rolling_vwap) / rolling_vwap.replace(0, np.nan)

    previous_close = features["close"].shift(1)
    true_range = pd.concat(
        [
            features["high"] - features["low"],
            (features["high"] - previous_close).abs(),
            (features["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    features["atr"] = true_range.rolling(14).mean()

    features["ema_fast"] = features["close"].ewm(span=12, adjust=False).mean()
    features["ema_slow"] = features["close"].ewm(span=26, adjust=False).mean()
    features["trend_strength"] = (features["ema_fast"] - features["ema_slow"]) / features["close"]
    features["momentum_20"] = features["close"].pct_change(20)
    features["momentum_63"] = features["close"].pct_change(63)
    features["momentum_126"] = features["close"].pct_change(126)
    features["volume_confirmation"] = features["volume"] / features["volume"].rolling(20).mean().replace(0, np.nan)
    features["close_zscore_20"] = (
        (features["close"] - features["close"].rolling(20).mean())
        / features["close"].rolling(20).std().replace(0, np.nan)
    )
    features["downside_vol_20"] = features["return"].clip(upper=0).rolling(20).std() * math.sqrt(252)
    features["drawdown_252"] = (features["close"] / features["close"].rolling(252).max()) - 1.0
    features["breakout_55"] = (features["close"] / features["high"].rolling(55).max()) - 1.0
    features["trend_persistence_20"] = (
        (features["close"].diff() > 0).rolling(20).mean()
    )
    features["support_gap_20"] = (features["close"] / features["low"].rolling(20).min()) - 1.0

    # --- NEW SOTA FEATURES ---

    # 1. RSI (14-period) — classic momentum oscillator
    features["rsi_14"] = _rsi(features["close"], period=14)

    # 2. ADX (14-period) — directional trend strength
    features["adx_14"] = _adx(features["high"], features["low"], features["close"], period=14)

    # 3. Choppiness Index — trending vs range-bound
    features["choppiness_14"] = _choppiness_index(features["high"], features["low"], features["close"], period=14)

    # 4. Garman-Klass Volatility — efficient estimator
    features["garman_klass_vol"] = _garman_klass_vol(
        features["high"], features["low"], features["close"], features["open"], window=20
    )

    # 5. Parkinson Volatility — high-low range based
    features["parkinson_vol"] = _parkinson_vol(features["high"], features["low"], window=20)

    # 6. MACD Histogram — momentum acceleration
    features["macd_histogram"] = _macd_histogram(features["close"])

    # 7. Delivery Percentage — India-specific institutional conviction
    # Falls back to 0.5 (neutral) when unavailable
    delivery_pct = alt.get("delivery_pct", None)
    if delivery_pct is not None and isinstance(delivery_pct, (int, float)) and delivery_pct > 0:
        features["delivery_pct"] = float(delivery_pct) / 100.0  # normalize to 0-1
    else:
        features["delivery_pct"] = 0.5  # neutral fallback

    # 8. Put-Call Ratio — massive directional signal for Indian markets
    pcr = alt.get("put_call_ratio", None)
    if pcr is not None and isinstance(pcr, (int, float)) and pcr > 0:
        features["put_call_ratio"] = float(pcr)
    else:
        features["put_call_ratio"] = 1.0  # neutral fallback

    # 9. India VIX Z-score — normalized fear gauge
    india_vix = alt.get("india_vix", None)
    if india_vix is not None and isinstance(india_vix, (int, float)) and india_vix > 0:
        # Use historical VIX mean ~15, std ~5 as baseline for Z-score
        features["india_vix_zscore"] = (float(india_vix) - 15.0) / 5.0
    else:
        features["india_vix_zscore"] = 0.0  # neutral fallback

    # 10. FII/DII Net Flow Direction — institutional money flow
    fii_dii_net = alt.get("fii_dii_net", None)
    if fii_dii_net is not None and isinstance(fii_dii_net, (int, float)):
        # Positive = net buying, negative = net selling, normalize by typical range
        features["fii_dii_signal"] = float(np.clip(fii_dii_net / 5000.0, -1.0, 1.0))
    else:
        features["fii_dii_signal"] = 0.0  # neutral fallback

    # --- Additional derived features ---
    # RSI divergence (price up but RSI down = bearish divergence)
    features["rsi_zscore"] = (features["rsi_14"] - 50.0) / 25.0  # normalized around neutral
    # Volatility regime ratio (GK vs realized)
    features["vol_regime_ratio"] = features["garman_klass_vol"] / features["realized_vol"].replace(0, np.nan)

    return features.replace([np.inf, -np.inf], np.nan).ffill().dropna()
