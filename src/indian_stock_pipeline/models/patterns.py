from __future__ import annotations

import math

import numpy as np
import pandas as pd

from indian_stock_pipeline.core.schemas import PatternSummary

try:
    import stumpy
except ImportError:  # pragma: no cover - dependency guarded for runtime flexibility
    stumpy = None


def _z_normalize(values: np.ndarray) -> np.ndarray:
    std = np.std(values)
    if std == 0:
        return np.zeros_like(values)
    return (values - np.mean(values)) / std


# ---------------------------------------------------------------------------
# FastDTW with Sakoe-Chiba band — O(n*radius) instead of O(n²)
# ---------------------------------------------------------------------------

def _fast_dtw(left: np.ndarray, right: np.ndarray, radius: int = 10) -> float:
    """DTW with Sakoe-Chiba band constraint for O(n*radius) complexity."""
    n, m = len(left), len(right)
    # Use a windowed approach — only compute cells within `radius` of diagonal
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - radius)
        j_end = min(m, i + radius)
        for j in range(j_start, j_end + 1):
            cost = abs(left[i - 1] - right[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / (n + m))


# ---------------------------------------------------------------------------
# Candlestick pattern detection — key reversal/continuation patterns
# ---------------------------------------------------------------------------

def _detect_candlestick_patterns(open_prices: pd.Series, high: pd.Series,
                                  low: pd.Series, close: pd.Series) -> list[str]:
    """Detect key candlestick patterns from the last 5 bars."""
    patterns = []
    if len(close) < 5:
        return patterns

    o = open_prices.iloc[-5:].values
    h = high.iloc[-5:].values
    l = low.iloc[-5:].values
    c = close.iloc[-5:].values

    body = c - o
    body_abs = np.abs(body)
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - l
    avg_body = np.mean(body_abs[:4])  # average of prior 4 bars

    # Latest bar
    b = body[-1]
    ba = body_abs[-1]
    us = upper_shadow[-1]
    ls = lower_shadow[-1]
    candle_range = h[-1] - l[-1]

    if candle_range == 0:
        return patterns

    # Doji — body < 10% of range
    if ba < 0.1 * candle_range:
        patterns.append("Doji")

    # Hammer — small body at top, long lower shadow (>2x body)
    if ls > 2.0 * ba and us < 0.3 * ba and b > 0 and body[-2] < 0:
        patterns.append("Hammer")

    # Inverted Hammer
    if us > 2.0 * ba and ls < 0.3 * ba and body[-2] < 0:
        patterns.append("Inverted Hammer")

    # Bullish Engulfing — current green candle engulfs prior red
    if body[-2] < 0 and b > 0 and o[-1] <= c[-2] and c[-1] >= o[-2]:
        patterns.append("Bullish Engulfing")

    # Bearish Engulfing — current red candle engulfs prior green
    if body[-2] > 0 and b < 0 and o[-1] >= c[-2] and c[-1] <= o[-2]:
        patterns.append("Bearish Engulfing")

    # Morning Star (3-bar) — big red, small body, big green
    if len(body) >= 3:
        if body[-3] < 0 and body_abs[-3] > avg_body and body_abs[-2] < avg_body * 0.3 and body[-1] > 0 and body_abs[-1] > avg_body:
            patterns.append("Morning Star")

    # Evening Star (3-bar)
    if len(body) >= 3:
        if body[-3] > 0 and body_abs[-3] > avg_body and body_abs[-2] < avg_body * 0.3 and body[-1] < 0 and body_abs[-1] > avg_body:
            patterns.append("Evening Star")

    # Shooting Star — small body at bottom, long upper shadow
    if us > 2.0 * ba and ls < 0.3 * ba and body[-2] > 0:
        patterns.append("Shooting Star")

    return patterns


class PatternMatcher:
    def __init__(self, window: int = 30):
        self.window = window

    def analyze(self, close: pd.Series, daily: pd.DataFrame | None = None) -> PatternSummary:
        clean = close.dropna()
        if len(clean) < self.window * 3:
            return PatternSummary(matrix_profile_score=0.5, dtw_similarity=0.5, anomaly_score=0.5)

        # Multi-window matrix profile
        multi_window_scores = self._multi_window_matrix_profile(clean)
        matrix_profile_score = float(np.mean(list(multi_window_scores.values()))) if multi_window_scores else 0.5
        anomaly_score = self._anomaly_from_mp(clean)

        # FastDTW similarity
        dtw_similarity = self._fast_dtw_similarity(clean)

        # Candlestick patterns
        candlestick_patterns = []
        if daily is not None and not daily.empty and len(daily) >= 5:
            candlestick_patterns = _detect_candlestick_patterns(
                daily["open"], daily["high"], daily["low"], daily["close"]
            )

        return PatternSummary(
            matrix_profile_score=matrix_profile_score,
            dtw_similarity=dtw_similarity,
            anomaly_score=anomaly_score,
            candlestick_patterns=candlestick_patterns,
            multi_window_mp_scores=multi_window_scores,
        )

    def _multi_window_matrix_profile(self, close: pd.Series) -> dict[int, float]:
        """Compute matrix profile at multiple windows for multi-scale pattern detection."""
        if stumpy is None:
            return {}

        scores = {}
        for w in [15, 30, 60]:
            if len(close) < w * 3:
                continue
            try:
                profile = stumpy.stump(close.to_numpy(dtype=float), m=w)
                distances = profile[:, 0].astype(float)
                finite = distances[np.isfinite(distances)]
                if len(finite) > 0:
                    latest = float(finite[-1])
                    scores[w] = float(np.clip(1.0 / (1.0 + max(latest, 0.0)), 0, 1))
            except Exception:
                continue
        return scores

    def _anomaly_from_mp(self, close: pd.Series) -> float:
        """Compute anomaly score from matrix profile of primary window."""
        if stumpy is None:
            return 0.5

        try:
            profile = stumpy.stump(close.to_numpy(dtype=float), m=self.window)
            distances = profile[:, 0].astype(float)
            finite = distances[np.isfinite(distances)]
            if len(finite) == 0:
                return 0.5
            latest = float(finite[-1])
            mean_d = float(np.mean(finite))
            std_d = float(np.std(finite)) or 1.0
            z = float(np.clip((latest - mean_d) / std_d, -3, 3))
            return float((z + 3.0) / 6.0)
        except Exception:
            return 0.5

    def _fast_dtw_similarity(self, close: pd.Series) -> float:
        """FastDTW with Sakoe-Chiba band for efficient similarity search."""
        recent = _z_normalize(close.iloc[-self.window:].to_numpy(dtype=float))
        best_distance = math.inf
        stride = max(3, self.window // 4)
        for start in range(0, len(close) - (2 * self.window), stride):
            historical_window = _z_normalize(close.iloc[start: start + self.window].to_numpy(dtype=float))
            distance = _fast_dtw(recent, historical_window, radius=10)
            best_distance = min(best_distance, distance)

        if not np.isfinite(best_distance):
            return 0.5
        return float(1.0 / (1.0 + best_distance))
