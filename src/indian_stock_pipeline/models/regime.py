from __future__ import annotations

import numpy as np
import pandas as pd

from indian_stock_pipeline.core.schemas import RegimeSummary

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - dependency guarded for runtime flexibility
    GaussianHMM = None

try:
    from pykalman import KalmanFilter
except ImportError:  # pragma: no cover - dependency guarded for runtime flexibility
    KalmanFilter = None

try:
    import ruptures as rpt
except ImportError:  # pragma: no cover - dependency guarded for runtime flexibility
    rpt = None


# ---------------------------------------------------------------------------
# Regime labels for human-readable output
# ---------------------------------------------------------------------------
_REGIME_LABELS = {
    "bull_low_vol": "Bullish / Low Volatility",
    "bull_high_vol": "Bullish / High Volatility",
    "bear_low_vol": "Bearish / Low Volatility",
    "bear_high_vol": "Bearish / High Volatility",
    "sideways": "Sideways / Range-bound",
    "unknown": "Unknown",
}


class RegimeDetector:
    def __init__(self, n_states: int = 3, persistence_days: int = 3):
        self.n_states = n_states
        self.persistence_days = persistence_days
        self._prev_state: int | None = None
        self._state_counter: int = 0

    def detect(self, features: pd.DataFrame) -> RegimeSummary:
        # Build regime input — now includes VIX and ADX if available
        base_cols = ["return", "realized_vol", "trend_strength"]
        extra_cols = []
        for col in ["adx_14", "choppiness_14", "india_vix_zscore", "fii_dii_signal"]:
            if col in features.columns:
                extra_cols.append(col)

        use_cols = base_cols + extra_cols
        regime_input = features[use_cols].replace([np.inf, -np.inf], np.nan).dropna()
        if regime_input.empty:
            raise ValueError("Not enough clean feature rows to run regime detection.")

        current_state = None
        trend_state = None
        state_probabilities: dict[int, float] = {}
        state_means: dict[int, float] = {}
        trending_probability = 0.5

        if GaussianHMM is not None and len(regime_input) >= 90:
            # Use full covariance when we have enough data, diagonal otherwise
            cov_type = "full" if len(regime_input) >= 250 else "diag"
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=cov_type,
                n_iter=500,
                random_state=42,
                tol=0.01,
            )
            train_values = regime_input.to_numpy()
            model.fit(train_values)
            predicted_states = model.predict(train_values)
            probabilities = model.predict_proba(train_values)

            state_means = {
                state: float(regime_input.loc[predicted_states == state, "return"].mean())
                for state in range(self.n_states)
            }
            trend_state = max(state_means, key=state_means.get)
            raw_state = int(predicted_states[-1])

            # --- Regime persistence filter ---
            # Require N consecutive days in a new state before transitioning
            if self._prev_state is None:
                self._prev_state = raw_state
                self._state_counter = self.persistence_days
            if raw_state == self._prev_state:
                self._state_counter = min(self._state_counter + 1, self.persistence_days * 2)
            else:
                self._state_counter -= 1
                if self._state_counter <= 0:
                    self._prev_state = raw_state
                    self._state_counter = self.persistence_days

            current_state = self._prev_state
            state_probabilities = {state: float(probabilities[-1][state]) for state in range(self.n_states)}
            trending_probability = float(state_probabilities.get(trend_state, 0.5))
        else:
            rolling_mean = regime_input["return"].rolling(20).mean()
            trend_state = 1
            current_state = int((rolling_mean.iloc[-1] > 0))
            state_probabilities = {0: float(1 - current_state), 1: float(current_state)}
            state_means = {0: float(regime_input["return"].mean() * -1), 1: float(regime_input["return"].mean())}
            trending_probability = float(max(state_probabilities.values()))

        kalman_price, kalman_target = self._kalman_prices(features["close"])
        breakpoints = self._breakpoints(features["close"])

        # --- Derive human-readable regime label ---
        regime_label = self._classify_regime(features, current_state, trend_state, state_means)

        # --- Extract VIX and FII/DII context ---
        vix_zscore = float(features["india_vix_zscore"].iloc[-1]) if "india_vix_zscore" in features.columns else 0.0
        fii_dii_signal = float(features["fii_dii_signal"].iloc[-1]) if "fii_dii_signal" in features.columns else 0.0
        fii_dii_direction = "buying" if fii_dii_signal > 0.1 else ("selling" if fii_dii_signal < -0.1 else "neutral")

        return RegimeSummary(
            current_state=current_state,
            trend_state=trend_state,
            trending_probability=trending_probability,
            kalman_price=kalman_price,
            kalman_target=kalman_target,
            breakpoints=breakpoints,
            state_probabilities=state_probabilities,
            state_return_means=state_means,
            regime_label=regime_label,
            vix_zscore=vix_zscore,
            fii_dii_direction=fii_dii_direction,
            regime_persistence_days=self._state_counter,
        )

    def _classify_regime(self, features: pd.DataFrame, current_state: int | None,
                         trend_state: int | None, state_means: dict[int, float]) -> str:
        """Derive a human-readable regime label from multiple signals."""
        latest = features.iloc[-1]
        trend = float(latest.get("trend_strength", 0))
        vol = float(latest.get("realized_vol", 0))
        adx = float(latest.get("adx_14", 25))
        chop = float(latest.get("choppiness_14", 50))
        vix_z = float(latest.get("india_vix_zscore", 0))

        # High vol threshold: VIX z-score > 1 or realized vol > 30%
        high_vol = vix_z > 1.0 or vol > 0.30
        # Trending: ADX > 25 and choppiness < 45
        is_trending = adx > 25 and chop < 45
        # Direction
        bullish = trend > 0 and current_state == trend_state

        if not is_trending:
            return "sideways"
        if bullish and not high_vol:
            return "bull_low_vol"
        if bullish and high_vol:
            return "bull_high_vol"
        if not bullish and not high_vol:
            return "bear_low_vol"
        if not bullish and high_vol:
            return "bear_high_vol"
        return "unknown"

    def _kalman_prices(self, close: pd.Series) -> tuple[float, float]:
        """2-state Kalman filter (price + velocity) for better trend projection."""
        clean = close.dropna()
        if clean.empty:
            return 0.0, 0.0

        if KalmanFilter is None or len(clean) < 20:
            smoothed = clean.ewm(span=10, adjust=False).mean()
            slope = smoothed.diff().tail(10).mean()
            latest = float(smoothed.iloc[-1])
            return latest, float(latest + slope * 5)

        # 2-state Kalman: [price, velocity]
        kalman = KalmanFilter(
            transition_matrices=[[1, 1], [0, 1]],
            observation_matrices=[[1, 0]],
            initial_state_mean=[clean.iloc[0], 0],
            initial_state_covariance=[[1, 0], [0, 1]],
            observation_covariance=[[1]],
            transition_covariance=[[0.01, 0], [0, 0.001]],
        )
        state_means, _ = kalman.filter(clean.values)
        smoothed_price = state_means[-1, 0]
        velocity = state_means[-1, 1]
        return float(smoothed_price), float(smoothed_price + velocity * 5)

    def _breakpoints(self, close: pd.Series) -> list[int]:
        clean = close.dropna()
        if rpt is None or len(clean) < 90:
            return []

        algo = rpt.Pelt(model="rbf").fit(clean.to_numpy())
        points = algo.predict(pen=8)
        return [int(point) for point in points[:-1]]
