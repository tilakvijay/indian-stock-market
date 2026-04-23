from __future__ import annotations

import os
from pathlib import Path

from indian_stock_pipeline.backtesting.vectorbt_runner import run_backtest
from indian_stock_pipeline.core.config import Settings
from indian_stock_pipeline.core.schemas import AnalysisResult
from indian_stock_pipeline.data.alternative import AlternativeDataCollector
from indian_stock_pipeline.data.providers import build_provider
from indian_stock_pipeline.data.resolver import CompanyResolver
from indian_stock_pipeline.features.engineering import compute_features
from indian_stock_pipeline.models.deep_forecaster import TorchForecaster
from indian_stock_pipeline.models.patterns import PatternMatcher
from indian_stock_pipeline.models.regime import RegimeDetector
from indian_stock_pipeline.models.screener import MarketScreener
from indian_stock_pipeline.models.signals import build_recommendation


def _extract_alternative_data_for_features(alternative_data) -> dict:
    """Extract relevant fields from AlternativeDataSummary for feature engineering."""
    alt = {}

    # Delivery percentage from bhavcopy
    if alternative_data.bhavcopy_delivery:
        dp = alternative_data.bhavcopy_delivery.get("delivery_percent")
        if dp is not None:
            try:
                alt["delivery_pct"] = float(dp)
            except (TypeError, ValueError):
                pass

    # Put-Call Ratio from option chain
    if alternative_data.option_chain:
        pcr = alternative_data.option_chain.get("put_call_ratio")
        if pcr is not None:
            try:
                alt["put_call_ratio"] = float(pcr)
            except (TypeError, ValueError):
                pass

    # India VIX from live quote or nsepython
    if alternative_data.live_quote:
        vix = alternative_data.live_quote.get("india_vix")
        if vix is not None:
            try:
                alt["india_vix"] = float(vix)
            except (TypeError, ValueError):
                pass

    # FII/DII net flow
    if alternative_data.fii_dii:
        try:
            # Sum net values from recent FII/DII entries
            net = 0.0
            for row in alternative_data.fii_dii[:5]:
                for key in ["fii_net", "FII_Net", "net_value", "Net Value", "FII/FPI Net"]:
                    val = row.get(key)
                    if val is not None:
                        try:
                            net += float(str(val).replace(",", ""))
                            break
                        except (TypeError, ValueError):
                            continue
            if net != 0:
                alt["fii_dii_net"] = net
        except Exception:
            pass

    return alt


class StockAnalysisPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._prepare_runtime_cache(settings)
        self.resolver = CompanyResolver(settings)
        self.alternative_data_collector = AlternativeDataCollector(settings)
        self.regime_detector = RegimeDetector(
            n_states=3,
            persistence_days=settings.regime_persistence_days,
        )
        self.pattern_matcher = PatternMatcher(window=30)
        self.deep_forecaster = TorchForecaster(settings)
        self.market_screener = MarketScreener(settings)

    def _prepare_runtime_cache(self, settings: Settings) -> None:
        market_cache = Path(settings.data_cache_dir).resolve()
        jugaad_cache = Path(settings.jugaad_cache_dir).resolve()
        market_cache.mkdir(parents=True, exist_ok=True)
        jugaad_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("J_CACHE_DIR", str(jugaad_cache))

    def analyze(self, company_query: str, include_screener: bool = False) -> AnalysisResult:
        instrument = self.resolver.resolve(company_query)
        provider = build_provider(self.settings, instrument)
        market_data = provider.fetch(instrument)
        alternative_data = self.alternative_data_collector.collect(instrument)

        # Extract alternative data for feature engineering
        alt_dict = _extract_alternative_data_for_features(alternative_data)

        # Compute features with alternative data integration
        features = compute_features(market_data.daily, market_data.intraday, alternative_data=alt_dict)

        # Regime detection (now uses VIX, ADX, choppiness, FII/DII)
        regime = self.regime_detector.detect(features)

        # Pattern matching (now includes candlestick patterns)
        patterns = self.pattern_matcher.analyze(features["close"], daily=features)

        # PatchTST deep forecaster
        forecast = self.deep_forecaster.forecast(features["close"], features_df=features)

        # Recommendation with regime-adaptive weights + Kelly sizing
        recommendation = build_recommendation(
            features, regime, patterns, forecast,
            kelly_cap=self.settings.kelly_fraction_cap,
        )

        # Walk-forward purged CV backtest
        backtest = run_backtest(
            features,
            settings=self.settings,
            regime_label=regime.regime_label,
            position_size=recommendation.position_size,
        )

        screener = self.market_screener.screen() if include_screener else None

        return AnalysisResult(
            instrument=instrument,
            market_data=market_data,
            alternative_data=alternative_data,
            features=features,
            regime=regime,
            patterns=patterns,
            forecast=forecast,
            recommendation=recommendation,
            backtest=backtest,
            screener=screener,
        )
