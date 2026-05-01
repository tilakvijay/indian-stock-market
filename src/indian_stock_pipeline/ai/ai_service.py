"""GLM 5.1 AI Service — NVIDIA NIM integration for intelligent stock analysis.

This module provides a high-performance wrapper around the GLM 5.1 model
hosted on NVIDIA's NIM platform. Designed for:
  - Market commentary generation
  - News-aware stock analysis (leverages model's internet capabilities)
  - Risk narrative synthesis
  - Interactive conversational analysis

Performance notes (observed):
  - ~71s full response, 11.65 TPS, 16ms TTFT
  - Reasoning is kept OFF for speed
  - Streaming is used for real-time output in Streamlit
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


@dataclass
class AIModelConfig:
    """Runtime-mutable AI model configuration."""
    api_key: str = ""
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model_name: str = "z-ai/glm-5.1"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    enable_thinking: bool = False
    clear_thinking: bool = True
    timeout: int = 120  # generous timeout — model is slow

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "enable_thinking": self.enable_thinking,
            "clear_thinking": self.clear_thinking,
            "timeout": self.timeout,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AIModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AIResponse:
    """Structured response from the AI model."""
    content: str
    model: str
    tokens_used: int = 0
    latency_seconds: float = 0.0
    tokens_per_second: float = 0.0
    error: str | None = None


class AIService:
    """High-performance GLM 5.1 client optimized for stock analysis."""

    def __init__(self, config: AIModelConfig | None = None):
        self.config = config or AIModelConfig()
        self._client = None

    def _get_client(self):
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                )
            except ImportError:
                raise ImportError("openai package is required. Install with: pip install openai")
        return self._client

    def update_config(self, config: AIModelConfig) -> None:
        """Hot-swap configuration without restarting."""
        self.config = config
        self._client = None  # Force re-initialization

    def is_configured(self) -> bool:
        """Check if API key is set."""
        return bool(self.config.api_key and self.config.api_key.strip())

    def _build_messages(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        context: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build message array with optional system prompt and conversation context."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _extra_body(self) -> dict[str, Any]:
        """Build extra_body with thinking controls for speed."""
        return {
            "chat_template_kwargs": {
                "enable_thinking": self.config.enable_thinking,
                "clear_thinking": self.config.clear_thinking,
            }
        }

    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        context: list[dict[str, str]] | None = None,
    ) -> AIResponse:
        """Non-streaming generation — waits for full response."""
        if not self.is_configured():
            return AIResponse(content="", model=self.config.model_name, error="API key not configured")

        client = self._get_client()
        messages = self._build_messages(user_prompt, system_prompt, context)

        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                extra_body=self._extra_body(),
                stream=False,
            )
            elapsed = time.perf_counter() - start
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else len(content.split())
            return AIResponse(
                content=content,
                model=self.config.model_name,
                tokens_used=tokens,
                latency_seconds=elapsed,
                tokens_per_second=tokens / elapsed if elapsed > 0 else 0,
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"AI generation failed: {e}")
            return AIResponse(
                content="",
                model=self.config.model_name,
                latency_seconds=elapsed,
                error=str(e),
            )

    def generate_stream(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        context: list[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        """Streaming generation — yields chunks as they arrive.

        This is the preferred method for Streamlit integration since the model
        is slow (~71s for full response) and streaming provides immediate feedback.
        """
        if not self.is_configured():
            yield "[Error: API key not configured. Go to ⚙️ AI Configuration in the sidebar.]"
            return

        client = self._get_client()
        messages = self._build_messages(user_prompt, system_prompt, context)

        try:
            completion = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                extra_body=self._extra_body(),
                stream=True,
            )
            for chunk in completion:
                if not getattr(chunk, "choices", None):
                    continue
                if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None) is not None:
                    yield delta.content
        except Exception as e:
            logger.error(f"AI streaming failed: {e}")
            yield f"\n\n[Error: {e}]"


# ---------------------------------------------------------------------------
# Pre-built prompt templates for stock analysis use-cases
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ANALYST = """You are an elite Indian stock market analyst with deep expertise in:
- NSE/BSE market microstructure, FII/DII flows, delivery percentages
- Technical analysis (RSI, ADX, MACD, Bollinger, support/resistance)
- Quantitative signals (Hurst exponent, regime detection, Kelly criterion)
- Indian macro-economy, RBI policy, sectoral rotation, India VIX
- Options flow analysis (put-call ratio, open interest)

Your analysis style:
- Be specific with numbers, levels, and actionable targets
- Mention current market news/events affecting the stock (use your internet knowledge)
- Always include risk factors and position sizing guidance
- Use a professional yet accessible tone
- Reference India-specific factors (monsoon impact, budget season, FII flows, etc.)
- Keep responses focused and structured with clear sections"""

SYSTEM_PROMPT_NEWS = """You are a real-time Indian financial news aggregator and analyst.
Search your knowledge for the LATEST news, developments, and market-moving events related to the query.
Focus on:
- Recent corporate actions, earnings, board meetings
- Regulatory changes (SEBI, RBI, government policy)
- Sectoral news and macro developments
- FII/DII activity patterns
- Global events impacting Indian markets

Format your response with clear categories and timestamps where possible.
Be specific about dates and sources. If you're unsure about recency, mention the knowledge cutoff."""

SYSTEM_PROMPT_RISK = """You are a risk management specialist for Indian equity markets.
Given the analysis data, provide a comprehensive risk assessment including:
- Key risk factors ranked by severity
- Correlation risks with broader market/sector
- Event risks (earnings, policy, global)
- Liquidity risks
- Position sizing recommendations
- Hedging suggestions using F&O if applicable
- Worst-case scenario analysis with specific price levels"""


def build_analysis_prompt(
    symbol: str,
    recommendation: Any,
    regime_label: str,
    latest_features: dict[str, Any],
    patterns: Any | None = None,
    forecast: Any | None = None,
) -> str:
    """Build a rich context prompt from pipeline analysis data."""
    lines = [
        f"## Stock Analysis Context for {symbol}",
        f"**Action**: {recommendation.action} | **Confidence**: {recommendation.confidence:.0%}",
        f"**Regime**: {regime_label.replace('_', ' ').title()}",
        f"**Setup Quality**: {recommendation.setup_quality}",
        "",
        "### Key Levels",
        f"- Entry zone: ₹{recommendation.entry_low:.2f} - ₹{recommendation.entry_high:.2f}",
        f"- Stop-loss: ₹{recommendation.stop_loss:.2f}",
        f"- Exit target: ₹{recommendation.exit_target:.2f}",
        f"- R:R ratio: {recommendation.risk_reward:.1f}",
        f"- Position size: {recommendation.position_size:.0%} | Kelly: {recommendation.kelly_fraction:.2%}",
        "",
        "### Latest Features",
    ]
    feature_keys = [
        "close", "atr", "momentum_20", "momentum_63", "rsi_14", "adx_14",
        "hurst", "trend_strength", "volume_confirmation", "ofi_proxy",
        "delivery_pct", "put_call_ratio", "realized_vol", "drawdown_252",
    ]
    for key in feature_keys:
        val = latest_features.get(key)
        if val is not None:
            if isinstance(val, float):
                lines.append(f"- {key}: {val:.4f}")
            else:
                lines.append(f"- {key}: {val}")

    if patterns:
        lines.extend([
            "",
            "### Pattern Analysis",
            f"- Matrix profile score: {patterns.matrix_profile_score:.2f}",
            f"- DTW similarity: {patterns.dtw_similarity:.2f}",
            f"- Anomaly score: {patterns.anomaly_score:.2f}",
        ])
        if patterns.candlestick_patterns:
            lines.append(f"- Candlestick patterns: {', '.join(patterns.candlestick_patterns)}")

    if forecast and forecast.enabled:
        lines.extend([
            "",
            "### Deep Model Forecast",
            f"- Predicted return: {forecast.predicted_return:.2%}",
            f"- Model confidence: {forecast.confidence:.2%}",
        ])

    if recommendation.risk_flags:
        lines.extend(["", "### Risk Flags"])
        for flag in recommendation.risk_flags:
            lines.append(f"- ⚠️ {flag}")

    lines.extend([
        "",
        "---",
        "Based on this quantitative analysis AND your knowledge of recent market news/events,",
        "provide a comprehensive, actionable analysis with specific recommendations.",
        "Include any recent news or developments that could impact this stock.",
        "Mention sector trends, FII/DII patterns, and macro factors relevant to this stock.",
    ])
    return "\n".join(lines)


def build_news_prompt(query: str) -> str:
    """Build a prompt for fetching latest news about a stock/topic."""
    return (
        f"Get me the latest and most impactful news related to: {query}\n\n"
        "Focus on:\n"
        "1. Recent corporate developments (last 1-4 weeks)\n"
        "2. Regulatory/policy changes affecting this stock/sector\n"
        "3. Analyst upgrades/downgrades and target price changes\n"
        "4. FII/DII activity in this stock or sector\n"
        "5. Global factors that could impact performance\n"
        "6. Upcoming events (earnings, dividends, board meetings)\n\n"
        "For each news item, mention the approximate date and potential market impact."
    )


def build_risk_prompt(
    symbol: str,
    recommendation: Any,
    regime_label: str,
    backtest: Any | None = None,
) -> str:
    """Build a prompt for deep risk analysis."""
    lines = [
        f"## Deep Risk Analysis for {symbol}",
        f"Current recommendation: {recommendation.action} at {recommendation.confidence:.0%} confidence",
        f"Market regime: {regime_label.replace('_', ' ').title()}",
        f"Position size: {recommendation.position_size:.0%}",
        f"Stop-loss: ₹{recommendation.stop_loss:.2f}",
    ]
    if recommendation.risk_flags:
        lines.append(f"\nExisting risk flags: {' | '.join(recommendation.risk_flags)}")
    if backtest and backtest.enabled:
        lines.extend([
            f"\nBacktest stats:",
            f"- Sharpe: {backtest.sharpe_ratio or 0:.2f}",
            f"- Max drawdown: {backtest.max_drawdown or 0:.2%}",
            f"- Win rate: {backtest.win_rate or 0:.2%}",
            f"- MC p-value: {backtest.monte_carlo_p_value or 1:.3f}",
        ])
    lines.extend([
        "",
        "Provide a comprehensive risk assessment with:",
        "1. Risk severity ranking (Critical / High / Medium / Low)",
        "2. Specific price levels for risk scenarios",
        "3. Hedging recommendations if applicable",
        "4. Position adjustment guidance",
        "5. Key events to watch in the next 2-4 weeks",
    ])
    return "\n".join(lines)
