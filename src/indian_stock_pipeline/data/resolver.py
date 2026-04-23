from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

import requests

from indian_stock_pipeline.core.config import Settings
from indian_stock_pipeline.core.schemas import ResolvedInstrument


ALLOWED_YAHOO_EXCHANGES = {"NSI", "BSE", "NSQ", "NSE"}


@dataclass(slots=True)
class CompanyResolver:
    settings: Settings

    def resolve(self, query: str) -> ResolvedInstrument:
        query = query.strip()
        if not query:
            raise ValueError("Please enter a company name or trading symbol.")

        direct = self._try_direct_symbol(query)
        if direct is not None:
            return direct

        yahoo_match = self._try_yahoo_search(query)
        if yahoo_match is not None:
            return yahoo_match

        fallback_symbol = query.upper().replace(" ", "") + (".BO" if self.settings.default_exchange == "BSE" else ".NS")
        return ResolvedInstrument(
            query=query,
            symbol=fallback_symbol,
            display_name=query.title(),
            exchange=self.settings.default_exchange,
            source="synthetic-fallback",
        )

    def _try_direct_symbol(self, query: str) -> ResolvedInstrument | None:
        upper = query.upper()
        if upper.endswith((".NS", ".BO")):
            exchange = "BSE" if upper.endswith(".BO") else "NSE"
            return ResolvedInstrument(
                query=query,
                symbol=upper,
                display_name=query.upper(),
                exchange=exchange,
                source="direct-input",
            )
        return None

    def _try_yahoo_search(self, query: str) -> ResolvedInstrument | None:
        try:
            response = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": query, "quotesCount": 10, "newsCount": 0, "lang": "en-US", "region": "IN"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=self.settings.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException:
            return None
        quotes = payload.get("quotes", [])
        if not quotes:
            return None

        scored: list[tuple[float, dict]] = []
        lowered_query = query.lower()
        for quote in quotes:
            symbol = quote.get("symbol")
            shortname = quote.get("shortname") or quote.get("longname") or ""
            exchange = quote.get("exchange") or quote.get("exchDisp") or ""
            if not symbol or exchange not in ALLOWED_YAHOO_EXCHANGES:
                continue
            score = max(
                SequenceMatcher(None, lowered_query, shortname.lower()).ratio(),
                SequenceMatcher(None, lowered_query, symbol.lower()).ratio(),
            )
            preferred_suffix = ".BO" if self.settings.default_exchange == "BSE" else ".NS"
            if str(symbol).upper().endswith(preferred_suffix):
                score += 0.08
            scored.append((score, quote))

        if not scored:
            return None

        best_score, best_quote = max(scored, key=lambda item: item[0])
        if best_score < 0.5:
            return None

        exchange = "BSE" if str(best_quote.get("symbol", "")).endswith(".BO") else "NSE"
        return ResolvedInstrument(
            query=query,
            symbol=best_quote["symbol"],
            display_name=best_quote.get("shortname") or best_quote.get("longname") or best_quote["symbol"],
            exchange=exchange,
            source="yahoo-search",
            metadata={"score": round(best_score, 3)},
        )
