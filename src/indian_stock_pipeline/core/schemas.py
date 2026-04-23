from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class ResolvedInstrument:
    query: str
    symbol: str
    display_name: str
    exchange: str
    source: str
    instrument_token: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MarketDataBundle:
    instrument: ResolvedInstrument
    daily: pd.DataFrame
    intraday: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AlternativeDataSummary:
    live_quote: dict[str, Any] = field(default_factory=dict)
    option_chain: dict[str, Any] = field(default_factory=dict)
    fii_dii: list[dict[str, Any]] = field(default_factory=list)
    bhavcopy_delivery: dict[str, Any] = field(default_factory=dict)
    bhavcopy_fo: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RegimeSummary:
    current_state: int | None
    trend_state: int | None
    trending_probability: float
    kalman_price: float
    kalman_target: float
    breakpoints: list[int]
    state_probabilities: dict[int, float]
    state_return_means: dict[int, float]
    regime_label: str = "unknown"
    vix_zscore: float = 0.0
    fii_dii_direction: str = "neutral"
    regime_persistence_days: int = 0


@dataclass(slots=True)
class PatternSummary:
    matrix_profile_score: float
    dtw_similarity: float
    anomaly_score: float
    candlestick_patterns: list[str] = field(default_factory=list)
    multi_window_mp_scores: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class ForecastSummary:
    predicted_return: float
    confidence: float
    device: str
    enabled: bool
    model_type: str = "disabled"
    training_loss: float = 0.0


@dataclass(slots=True)
class FoldMetric:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int


@dataclass(slots=True)
class BacktestSummary:
    enabled: bool
    total_return: float | None = None
    benchmark_return: float | None = None
    annualized_return: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    total_trades: int | None = None
    expectancy: float | None = None
    avg_trade_return: float | None = None
    last_signal: str | None = None
    trade_log: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: str | None = None
    fold_metrics: list[FoldMetric] = field(default_factory=list)
    monte_carlo_p_value: float | None = None
    monte_carlo_median_return: float | None = None


@dataclass(slots=True)
class Recommendation:
    action: str
    confidence: float
    entry_low: float
    entry_high: float
    stop_loss: float
    trailing_stop: float
    exit_target: float
    risk_reward: float
    notes: list[str]
    score_breakdown: dict[str, float]
    headline: str
    summary: str
    horizon: str
    risk_flags: list[str]
    setup_quality: str
    feature_explanations: dict[str, str]
    position_size: float = 1.0
    kelly_fraction: float = 0.0
    regime_label: str = "unknown"


@dataclass(slots=True)
class ScreenIdea:
    symbol: str
    display_name: str
    action: str
    confidence: float
    setup_quality: str
    latest_close: float
    momentum_20: float
    momentum_63: float
    trend_strength: float
    risk_reward: float
    summary: str


@dataclass(slots=True)
class ScreenResult:
    universe_name: str
    scanned_count: int
    long_candidates: list[ScreenIdea] = field(default_factory=list)
    short_candidates: list[ScreenIdea] = field(default_factory=list)
    skipped_symbols: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnalysisResult:
    instrument: ResolvedInstrument
    market_data: MarketDataBundle
    alternative_data: AlternativeDataSummary
    features: pd.DataFrame
    regime: RegimeSummary
    patterns: PatternSummary
    forecast: ForecastSummary
    recommendation: Recommendation
    backtest: BacktestSummary
    screener: ScreenResult | None = None
