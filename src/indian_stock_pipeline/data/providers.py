from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from indian_stock_pipeline.core.config import Settings
from indian_stock_pipeline.core.schemas import MarketDataBundle, ResolvedInstrument

try:
    from jugaad_data.nse import stock_df
except ImportError:  # pragma: no cover - dependency guarded for runtime flexibility
    stock_df = None


STANDARD_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "VOLUME": "volume",
    "DATE": "date",
    "TIMESTAMP": "date",
}


def _parse_history_period(period: str) -> int:
    normalized = period.strip().lower()
    if normalized.endswith("y"):
        return int(normalized[:-1]) * 365
    if normalized.endswith("mo"):
        return int(normalized[:-2]) * 30
    if normalized.endswith("d"):
        return int(normalized[:-1])
    return 730


def _normalize_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    normalized = frame.rename(columns=STANDARD_COLUMNS).copy()
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"])
        normalized = normalized.set_index("date")

    index = pd.to_datetime(normalized.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    normalized.index = index
    for column in ["open", "high", "low", "close", "volume"]:
        if column not in normalized:
            normalized[column] = 0.0
    normalized["volume"] = pd.to_numeric(normalized["volume"], errors="coerce").fillna(0.0)
    return normalized[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna(subset=["close"]).sort_index()


def _symbol_root(symbol: str) -> str:
    return symbol.split(".")[0].upper()


@dataclass(slots=True)
class YahooFinanceProvider:
    settings: Settings

    def fetch(self, instrument: ResolvedInstrument, include_intraday: bool = True) -> MarketDataBundle:
        ticker = yf.Ticker(instrument.symbol)
        daily = ticker.history(period=self.settings.history_period, interval=self.settings.bar_interval, auto_adjust=False)
        intraday = pd.DataFrame()
        if include_intraday:
            intraday = ticker.history(
                period=f"{max(self.settings.intraday_lookback_days, 7)}d",
                interval=self.settings.intraday_interval,
                auto_adjust=False,
            )
        return MarketDataBundle(
            instrument=instrument,
            daily=_normalize_history_frame(daily),
            intraday=_normalize_history_frame(intraday),
            metadata={"daily_source": "yfinance", "intraday_source": "yfinance"},
        )


@dataclass(slots=True)
class JugaadDataProvider:
    settings: Settings

    def fetch(self, instrument: ResolvedInstrument, include_intraday: bool = True) -> MarketDataBundle:
        if stock_df is None:
            raise RuntimeError("jugaad-data is not installed in the runtime environment.")

        try:
            daily = self._fetch_jugaad_daily(instrument)
            intraday = self._fetch_yfinance_intraday(instrument) if include_intraday else pd.DataFrame()
            metadata = {
                "daily_source": "jugaad-data",
                "intraday_source": "yfinance" if include_intraday else "disabled",
                "cross_validation": self._cross_validate_latest_close(instrument, daily),
            }
        except Exception as exc:
            fallback = YahooFinanceProvider(self.settings).fetch(instrument, include_intraday=include_intraday)
            fallback.metadata["daily_source"] = "yfinance-fallback"
            fallback.metadata["fallback_reason"] = str(exc)
            return fallback

        if daily.empty:
            fallback = YahooFinanceProvider(self.settings).fetch(instrument, include_intraday=include_intraday)
            fallback.metadata["daily_source"] = "yfinance-fallback"
            fallback.metadata["intraday_source"] = "yfinance" if include_intraday else "disabled"
            return fallback

        return MarketDataBundle(
            instrument=instrument,
            daily=daily,
            intraday=intraday,
            metadata=metadata,
        )

    def _fetch_jugaad_daily(self, instrument: ResolvedInstrument) -> pd.DataFrame:
        lookback_days = _parse_history_period(self.settings.history_period)
        start = date.today() - timedelta(days=lookback_days)
        end = date.today()
        frame = stock_df(symbol=_symbol_root(instrument.symbol), from_date=start, to_date=end, series="EQ")
        return _normalize_history_frame(frame)

    def _fetch_yfinance_intraday(self, instrument: ResolvedInstrument) -> pd.DataFrame:
        ticker = yf.Ticker(instrument.symbol)
        intraday = ticker.history(
            period=f"{max(self.settings.intraday_lookback_days, 7)}d",
            interval=self.settings.intraday_interval,
            auto_adjust=False,
        )
        return _normalize_history_frame(intraday)

    def _cross_validate_latest_close(self, instrument: ResolvedInstrument, daily: pd.DataFrame) -> dict[str, float] | dict[str, str]:
        if daily.empty:
            return {"status": "missing-jugaad-daily"}

        ticker = yf.Ticker(instrument.symbol)
        yahoo_daily = _normalize_history_frame(
            ticker.history(period="1mo", interval=self.settings.bar_interval, auto_adjust=False)
        )
        if yahoo_daily.empty:
            return {"status": "missing-yfinance-cross-check"}

        jugaad_close = float(daily["close"].iloc[-1])
        yahoo_close = float(yahoo_daily["close"].iloc[-1])
        diff_pct = ((jugaad_close - yahoo_close) / yahoo_close) if yahoo_close else 0.0
        return {
            "jugaad_close": round(jugaad_close, 4),
            "yfinance_close": round(yahoo_close, 4),
            "diff_pct": round(diff_pct, 6),
        }


def build_provider(settings: Settings, instrument: ResolvedInstrument):
    if settings.data_provider == "yfinance":
        return YahooFinanceProvider(settings)
    if settings.data_provider in {"auto", "jugaad"}:
        return JugaadDataProvider(settings)
    return YahooFinanceProvider(settings)
