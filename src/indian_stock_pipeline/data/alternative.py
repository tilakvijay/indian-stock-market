from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from indian_stock_pipeline.core.config import Settings
from indian_stock_pipeline.core.schemas import AlternativeDataSummary, ResolvedInstrument

try:
    import nsepython as nsep
except ImportError:  # pragma: no cover - dependency guarded for runtime flexibility
    try:
        import nsepythonserver as nsep
    except ImportError:  # pragma: no cover - dependency guarded for runtime flexibility
        nsep = None

try:
    from jugaad_data.nse import NSELive, bhavcopy_fo_save, bhavcopy_save
except ImportError:  # pragma: no cover - dependency guarded for runtime flexibility
    NSELive = None
    bhavcopy_fo_save = None
    bhavcopy_save = None


def _symbol_root(symbol: str) -> str:
    return symbol.split(".")[0].upper()


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


@dataclass(slots=True)
class AlternativeDataCollector:
    settings: Settings

    def collect(self, instrument: ResolvedInstrument) -> AlternativeDataSummary:
        symbol = _symbol_root(instrument.symbol)
        notes: list[str] = []

        live_quote = self._collect_live_quote(symbol)
        if not live_quote:
            notes.append("Live quote snapshot is unavailable in the current runtime.")

        option_chain = self._collect_option_chain(symbol)
        if not option_chain:
            notes.append("nsepython option-chain snapshot was not available for this symbol.")

        fii_dii = self._collect_fii_dii()
        if not fii_dii:
            notes.append("FII/DII flow snapshot could not be loaded from nsepython.")

        bhavcopy_delivery = self._collect_delivery_bhavcopy(symbol)
        if not bhavcopy_delivery:
            notes.append("Delivery bhavcopy summary could not be extracted from jugaad-data.")

        bhavcopy_fo = self._collect_fo_bhavcopy(symbol)
        if not bhavcopy_fo:
            notes.append("F&O bhavcopy OI summary could not be extracted from jugaad-data.")

        return AlternativeDataSummary(
            live_quote=live_quote,
            option_chain=option_chain,
            fii_dii=fii_dii,
            bhavcopy_delivery=bhavcopy_delivery,
            bhavcopy_fo=bhavcopy_fo,
            notes=notes,
        )

    def _collect_live_quote(self, symbol: str) -> dict[str, Any]:
        if NSELive is None:
            return {}

        live = NSELive()
        payload = _safe_call(live.stock_quote, symbol)
        if not payload:
            return {}

        price_info = payload.get("priceInfo", {})
        security_info = payload.get("securityInfo", {})
        return {
            "last_price": price_info.get("lastPrice"),
            "open": price_info.get("open"),
            "day_high": price_info.get("intraDayHighLow", {}).get("max"),
            "day_low": price_info.get("intraDayHighLow", {}).get("min"),
            "vwap": price_info.get("vwap"),
            "delivery_to_traded_quantity": security_info.get("deliveredQuantity"),
        }

    def _collect_option_chain(self, symbol: str) -> dict[str, Any]:
        if nsep is None:
            return {}

        if hasattr(nsep, "oi_chain_builder"):
            payload = _safe_call(nsep.oi_chain_builder, symbol, expiry="latest", oi_mode="compact")
            if payload and isinstance(payload, tuple) and len(payload) >= 3:
                chain_frame, underlying_value, timestamp = payload[:3]
                return self._summarize_option_chain_frame(chain_frame, underlying_value, timestamp)

        for candidate_name in ["option_chain", "nse_optionchain_scrapper"]:
            candidate = getattr(nsep, candidate_name, None)
            if callable(candidate):
                payload = _safe_call(candidate, symbol)
                if payload:
                    return self._summarize_option_chain_payload(payload)

        return {}

    def _summarize_option_chain_frame(self, frame: Any, underlying_value: Any, timestamp: Any) -> dict[str, Any]:
        if frame is None:
            return {}

        if not isinstance(frame, pd.DataFrame):
            try:
                frame = pd.DataFrame(frame)
            except Exception:
                return {}

        if frame.empty:
            return {}

        normalized = frame.copy()
        columns = {column.lower(): column for column in normalized.columns}
        strike_col = columns.get("strike price", columns.get("strike_price"))
        call_oi_col = columns.get("calls_oi")
        put_oi_col = columns.get("puts_oi")
        if not strike_col or not call_oi_col or not put_oi_col:
            return {"underlying_value": underlying_value, "timestamp": str(timestamp), "rows": int(len(normalized))}

        normalized[call_oi_col] = pd.to_numeric(normalized[call_oi_col], errors="coerce").fillna(0.0)
        normalized[put_oi_col] = pd.to_numeric(normalized[put_oi_col], errors="coerce").fillna(0.0)
        normalized[strike_col] = pd.to_numeric(normalized[strike_col], errors="coerce")

        max_call = normalized.loc[normalized[call_oi_col].idxmax()]
        max_put = normalized.loc[normalized[put_oi_col].idxmax()]
        total_call = float(normalized[call_oi_col].sum())
        total_put = float(normalized[put_oi_col].sum())

        return {
            "underlying_value": underlying_value,
            "timestamp": str(timestamp),
            "max_call_oi_strike": float(max_call[strike_col]),
            "max_put_oi_strike": float(max_put[strike_col]),
            "put_call_ratio": round(total_put / total_call, 4) if total_call else None,
            "rows": int(len(normalized)),
        }

    def _summarize_option_chain_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        records = payload.get("records", {})
        data = records.get("data", [])
        if not data:
            return {}

        rows = []
        for item in data:
            strike = item.get("strikePrice")
            ce = item.get("CE", {})
            pe = item.get("PE", {})
            rows.append(
                {
                    "strike": strike,
                    "call_oi": ce.get("openInterest", 0),
                    "put_oi": pe.get("openInterest", 0),
                }
            )
        frame = pd.DataFrame(rows)
        return self._summarize_option_chain_frame(frame.rename(columns={"strike": "Strike Price", "call_oi": "CALLS_OI", "put_oi": "PUTS_OI"}), records.get("underlyingValue"), records.get("timestamp"))

    def _collect_fii_dii(self) -> list[dict[str, Any]]:
        if nsep is None:
            return []

        for candidate_name in ["fii_dii_data", "fii_dii", "nse_fiidii"]:
            candidate = getattr(nsep, candidate_name, None)
            if callable(candidate):
                payload = _safe_call(candidate)
                rows = self._rows_from_payload(payload)
                if rows:
                    return rows[:5]
        return []

    def _collect_delivery_bhavcopy(self, symbol: str) -> dict[str, Any]:
        if bhavcopy_save is None:
            return {}

        cache_dir = Path(self.settings.data_cache_dir) / "bhavcopy_eq"
        frame = self._download_latest_bhavcopy(cache_dir, bhavcopy_save)
        if frame.empty:
            return {}

        normalized = self._uppercase_columns(frame)
        if "SYMBOL" not in normalized.columns:
            return {}
        symbol_row = normalized.loc[normalized["SYMBOL"].astype(str).str.upper() == symbol]
        if symbol_row.empty:
            return {}
        row = symbol_row.iloc[-1]
        return {
            "date": str(row.get("DATE1") or row.get("TIMESTAMP") or ""),
            "close": row.get("CLOSE"),
            "traded_quantity": row.get("TOTTRDQTY"),
            "delivery_quantity": row.get("DELIV_QTY"),
            "delivery_percent": row.get("DELIV_PER"),
        }

    def _collect_fo_bhavcopy(self, symbol: str) -> dict[str, Any]:
        if bhavcopy_fo_save is None:
            return {}

        cache_dir = Path(self.settings.data_cache_dir) / "bhavcopy_fo"
        frame = self._download_latest_bhavcopy(cache_dir, bhavcopy_fo_save)
        if frame.empty:
            return {}

        normalized = self._uppercase_columns(frame)
        if "SYMBOL" not in normalized.columns:
            return {}
        symbol_row = normalized.loc[normalized["SYMBOL"].astype(str).str.upper() == symbol]
        if symbol_row.empty:
            return {}
        sort_columns = [column for column in ["EXPIRY_DT", "TIMESTAMP"] if column in symbol_row.columns]
        row = symbol_row.sort_values(by=sort_columns).iloc[-1] if sort_columns else symbol_row.iloc[-1]
        return {
            "instrument": row.get("INSTRUMENT"),
            "expiry": row.get("EXPIRY_DT"),
            "open_interest": row.get("OPEN_INT"),
            "change_in_oi": row.get("CHG_IN_OI"),
            "settle_price": row.get("SETTLE_PR"),
        }

    def _download_latest_bhavcopy(self, cache_dir: Path, downloader) -> pd.DataFrame:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for lookback in range(0, 7):
            current_date = date.today() - timedelta(days=lookback)
            _safe_call(downloader, current_date, str(cache_dir))
        candidates = sorted(cache_dir.glob("*"), reverse=True)
        for candidate in candidates:
            if candidate.is_file():
                try:
                    return pd.read_csv(candidate)
                except Exception:
                    continue
        return pd.DataFrame()

    def _rows_from_payload(self, payload: Any) -> list[dict[str, Any]]:
        if payload is None:
            return []
        if isinstance(payload, pd.DataFrame):
            return payload.to_dict(orient="records")
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ["data", "value", "rows"]:
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
            return [payload]
        return []

    def _uppercase_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized.columns = [str(column).upper() for column in normalized.columns]
        return normalized
