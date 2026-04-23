from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indian_stock_pipeline.core.config import Settings
from indian_stock_pipeline.core.schemas import ResolvedInstrument, ScreenIdea, ScreenResult
from indian_stock_pipeline.data.providers import build_provider
from indian_stock_pipeline.features.engineering import compute_features
from indian_stock_pipeline.models.signals import build_signal_frame

try:
    from jugaad_data.nse import NSELive
except ImportError:  # pragma: no cover - dependency guarded for runtime flexibility
    NSELive = None


FALLBACK_UNIVERSE = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "ITC",
    "LT",
    "SBIN",
    "BHARTIARTL",
    "AXISBANK",
    "KOTAKBANK",
    "HINDUNILVR",
    "SUNPHARMA",
    "TATAMOTORS",
    "MARUTI",
    "NTPC",
    "POWERGRID",
    "BAJFINANCE",
    "ULTRACEMCO",
    "ADANIPORTS",
    "TITAN",
    "M&M",
    "WIPRO",
    "TECHM",
    "ASIANPAINT",
]


def _bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(np.clip(value, lower, upper))


@dataclass(slots=True)
class MarketScreener:
    settings: Settings

    def screen(self, universe_name: str = "NIFTY 50", limit: int = 5) -> ScreenResult:
        symbols, notes = self._resolve_universe(universe_name)
        symbols = symbols[: max(limit * 3, self.settings.screener_universe_size)]

        long_ideas: list[tuple[float, ScreenIdea]] = []
        short_ideas: list[tuple[float, ScreenIdea]] = []
        skipped_symbols: list[str] = []

        for symbol in symbols:
            instrument = ResolvedInstrument(
                query=symbol,
                symbol=f"{symbol}.NS",
                display_name=symbol,
                exchange="NSE",
                source="market-screener",
            )
            try:
                provider = build_provider(self.settings, instrument)
                market_data = provider.fetch(instrument, include_intraday=False)
                features = compute_features(market_data.daily, pd.DataFrame())
                if len(features) < 160:
                    skipped_symbols.append(symbol)
                    continue
            except Exception:
                skipped_symbols.append(symbol)
                continue

            latest = features.iloc[-1]
            signal_row = build_signal_frame(features, regime_label="unknown").iloc[-1]
            long_score = (
                0.45 * float(signal_row["score"])
                + 0.20 * float(signal_row["regime"])
                + 0.20 * _bounded((float(latest["momentum_126"]) + 0.35) / 0.70)
                + 0.15 * _bounded((float(latest["breakout_55"]) + 0.08) / 0.16)
            )
            short_score = (
                0.40 * (1.0 - float(signal_row["score"]))
                + 0.20 * _bounded((-float(latest["trend_strength"]) + 0.04) / 0.08)
                + 0.20 * _bounded((-float(latest["momentum_20"]) + 0.12) / 0.24)
                + 0.20 * _bounded((-float(latest["drawdown_252"])) / 0.35)
            )

            long_ideas.append(
                (
                    long_score,
                    ScreenIdea(
                        symbol=instrument.symbol,
                        display_name=instrument.display_name,
                        action="BUY" if long_score >= 0.62 else "WATCH",
                        confidence=float(_bounded(long_score)),
                        setup_quality="High" if long_score >= 0.70 else ("Developing" if long_score >= 0.58 else "Average"),
                        latest_close=float(latest["close"]),
                        momentum_20=float(latest["momentum_20"]),
                        momentum_63=float(latest["momentum_63"]),
                        trend_strength=float(latest["trend_strength"]),
                        risk_reward=max((float(latest["close"]) + (2.2 * float(latest["atr"])) - float(latest["close"])) / max(2.0 * float(latest["atr"]), 1e-6), 0.0),
                        summary=self._build_long_summary(latest, signal_row),
                    ),
                )
            )
            short_ideas.append(
                (
                    short_score,
                    ScreenIdea(
                        symbol=instrument.symbol,
                        display_name=instrument.display_name,
                        action="SELL" if short_score >= 0.62 else "AVOID",
                        confidence=float(_bounded(short_score)),
                        setup_quality="High" if short_score >= 0.70 else ("Developing" if short_score >= 0.58 else "Average"),
                        latest_close=float(latest["close"]),
                        momentum_20=float(latest["momentum_20"]),
                        momentum_63=float(latest["momentum_63"]),
                        trend_strength=float(latest["trend_strength"]),
                        risk_reward=1.0,
                        summary=self._build_short_summary(latest, signal_row),
                    ),
                )
            )

        long_candidates = [idea for _, idea in sorted(long_ideas, key=lambda item: item[0], reverse=True)[:limit]]
        short_candidates = [idea for _, idea in sorted(short_ideas, key=lambda item: item[0], reverse=True)[:limit]]

        if skipped_symbols:
            notes.append(f"Skipped {len(skipped_symbols)} symbols because data was incomplete or unavailable.")

        return ScreenResult(
            universe_name=universe_name,
            scanned_count=len(long_ideas),
            long_candidates=long_candidates,
            short_candidates=short_candidates,
            skipped_symbols=skipped_symbols,
            notes=notes,
        )

    def _resolve_universe(self, universe_name: str) -> tuple[list[str], list[str]]:
        notes: list[str] = []
        if NSELive is not None:
            try:
                live = NSELive()
                payload = live.live_index(universe_name)
                data_rows = payload.get("data", []) if isinstance(payload, dict) else []
                parsed_symbols = []
                for row in data_rows:
                    symbol = row.get("symbol") or row.get("meta", {}).get("symbol")
                    if symbol:
                        parsed_symbols.append(str(symbol).upper())
                if parsed_symbols:
                    return parsed_symbols, notes
            except Exception:
                notes.append(f"Could not load live universe '{universe_name}', so the screener used a liquid fallback list.")

        notes.append("Screener universe is a liquid large-cap fallback basket for speed and reliability.")
        return FALLBACK_UNIVERSE, notes

    def _build_long_summary(self, latest: pd.Series, signal_row: pd.Series) -> str:
        return (
            f"Trend strength {latest['trend_strength']:.2%}, 3-month momentum {latest['momentum_63']:.2%}, "
            f"and composite score {signal_row['score']:.0%} favor upside continuation."
        )

    def _build_short_summary(self, latest: pd.Series, signal_row: pd.Series) -> str:
        return (
            f"Trend strength {latest['trend_strength']:.2%}, 1-month momentum {latest['momentum_20']:.2%}, "
            f"and composite score {signal_row['score']:.0%} point to weak tactical structure."
        )
