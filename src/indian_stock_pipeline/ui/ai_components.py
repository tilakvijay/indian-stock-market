"""AI-powered UI components for Streamlit — GLM 5.1 integration.

Key fix: All AI actions store their output in session_state so they
persist across Streamlit reruns (buttons trigger reruns by default).
"""
from __future__ import annotations
import streamlit as st
from indian_stock_pipeline.ai.ai_service import (
    AIModelConfig, AIService,
    SYSTEM_PROMPT_ANALYST, SYSTEM_PROMPT_NEWS, SYSTEM_PROMPT_RISK,
    build_analysis_prompt, build_news_prompt, build_risk_prompt,
)


def _get_ai_service() -> AIService:
    if "ai_service" not in st.session_state:
        st.session_state.ai_service = AIService()
    return st.session_state.ai_service


def _get_ai_config() -> AIModelConfig:
    if "ai_config" not in st.session_state:
        st.session_state.ai_config = AIModelConfig()
    return st.session_state.ai_config


def render_ai_config_sidebar(settings) -> None:
    """Render AI configuration controls in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Configuration")

    cfg = _get_ai_config()

    api_key = st.sidebar.text_input(
        "API Key (NVIDIA NIM)",
        value=cfg.api_key or settings.ai_api_key,
        type="password",
        key="ai_api_key_input",
    )
    model_name = st.sidebar.text_input(
        "Model Name",
        value=cfg.model_name if cfg.model_name != "z-ai/glm-5.1" else settings.ai_model_name,
        key="ai_model_input",
    )
    base_url = st.sidebar.text_input(
        "Base URL",
        value=cfg.base_url,
        key="ai_base_url_input",
    )

    col1, col2 = st.sidebar.columns(2)
    temperature = col1.slider("Temperature", 0.0, 2.0, cfg.temperature, 0.1, key="ai_temp")
    top_p = col2.slider("Top-P", 0.0, 1.0, cfg.top_p, 0.05, key="ai_top_p")
    max_tokens = st.sidebar.slider("Max Tokens", 512, 16384, cfg.max_tokens, 256, key="ai_max_tok")
    enable_thinking = st.sidebar.checkbox("Enable Reasoning (slower)", value=cfg.enable_thinking, key="ai_think")

    new_cfg = AIModelConfig(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        clear_thinking=True,
        timeout=settings.ai_timeout,
    )
    st.session_state.ai_config = new_cfg
    svc = _get_ai_service()
    svc.update_config(new_cfg)

    if new_cfg.api_key:
        st.sidebar.success("✅ AI connected")
    else:
        st.sidebar.warning("⚠️ Enter API key to enable AI")


def render_ai_commentary(result) -> None:
    """Render AI-generated market commentary for a stock analysis result.

    Uses session_state to persist AI outputs across Streamlit reruns so that
    clicking a button doesn't reset the page.
    """
    svc = _get_ai_service()
    if not svc.is_configured():
        st.info("💡 Configure your NVIDIA NIM API key in the sidebar to unlock AI-powered analysis.")
        return

    symbol = result.instrument.display_name
    latest = result.features.iloc[-1]
    latest_dict = {k: (float(v) if hasattr(v, '__float__') else v) for k, v in latest.items()}

    st.subheader("🧠 AI Market Commentary")

    # --- Use session_state keys to track which action to run ---
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Deep Analysis", key="ai_analysis_btn", use_container_width=True):
            st.session_state["ai_action"] = "analysis"
            st.session_state.pop("ai_result_cache", None)
    with col2:
        if st.button("📰 Latest News", key="ai_news_btn", use_container_width=True):
            st.session_state["ai_action"] = "news"
            st.session_state.pop("ai_result_cache", None)
    with col3:
        if st.button("⚠️ Risk Report", key="ai_risk_btn", use_container_width=True):
            st.session_state["ai_action"] = "risk"
            st.session_state.pop("ai_result_cache", None)

    action = st.session_state.get("ai_action")
    cached = st.session_state.get("ai_result_cache")

    # If we have a cached result for the current action, show it
    if cached and cached.get("action") == action:
        st.markdown(cached["content"])
        return

    # Otherwise, stream the response
    if action == "analysis":
        prompt = build_analysis_prompt(
            symbol, result.recommendation, result.regime.regime_label,
            latest_dict, result.patterns, result.forecast,
        )
        _stream_ai(svc, prompt, SYSTEM_PROMPT_ANALYST, action, "🔄 GLM 5.1 analyzing (this takes ~60-90s)...")

    elif action == "news":
        prompt = build_news_prompt(f"{symbol} Indian stock market")
        _stream_ai(svc, prompt, SYSTEM_PROMPT_NEWS, action, "🔄 Fetching latest news via GLM 5.1...")

    elif action == "risk":
        prompt = build_risk_prompt(
            symbol, result.recommendation,
            result.regime.regime_label, result.backtest,
        )
        _stream_ai(svc, prompt, SYSTEM_PROMPT_RISK, action, "🔄 Generating risk report...")


def _stream_ai(svc: AIService, prompt: str, system_prompt: str, action: str, spinner_msg: str) -> None:
    """Stream AI response and cache it in session_state."""
    with st.spinner(spinner_msg):
        container = st.empty()
        full = ""
        for chunk in svc.generate_stream(prompt, system_prompt):
            full += chunk
            container.markdown(full)
    # Cache the result so it persists on next rerun
    st.session_state["ai_result_cache"] = {"action": action, "content": full}


def render_ai_chat(result=None) -> None:
    """Render an interactive AI chat interface."""
    svc = _get_ai_service()
    if not svc.is_configured():
        st.info("💡 Configure your NVIDIA NIM API key in the sidebar to enable AI chat.")
        return

    st.subheader("💬 AI Stock Analyst Chat")
    st.caption("Ask anything about Indian markets — the model has internet-aware knowledge.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Show history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about any stock, sector, or market trend...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add analysis context if available
        context_msgs = []
        if result:
            ctx = (
                f"[Context: Analyzing {result.instrument.display_name}, "
                f"Action={result.recommendation.action}, "
                f"Confidence={result.recommendation.confidence:.0%}, "
                f"Regime={result.regime.regime_label}]"
            )
            context_msgs = [{"role": "system", "content": ctx}]

        with st.chat_message("assistant"):
            container = st.empty()
            full = ""
            for chunk in svc.generate_stream(
                user_input, SYSTEM_PROMPT_ANALYST,
                context=context_msgs + st.session_state.chat_history[:-1],
            ):
                full += chunk
                container.markdown(full)

        st.session_state.chat_history.append({"role": "assistant", "content": full})

    if st.session_state.chat_history and st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
