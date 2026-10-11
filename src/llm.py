"""LLM 客户端工厂：统一从 .env + config 构建 ChatOpenAI，避免各处硬编码。

优先级：显式参数 > .env（DEEPSEEK_BASE_URL / DEEPSEEK_MODEL）> 默认值
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_BASE_URL: str = "https://api.deepseek.com"
DEFAULT_MODEL: str = "deepseek-chat"


def get_chat_model(
    temperature: float = 0.1,
    max_tokens: int | None = None,
    config: dict[str, Any] | None = None,
    json_mode: bool = False,
) -> ChatOpenAI:
    """构建聊天模型客户端。

    Args:
        temperature: 采样温度；judge/打分类调用应传 0
        max_tokens: 最大生成 token 数
        config: 配置 dict（agent.llm_model / agent.llm_base_url 可覆盖）
        json_mode: 是否强制 JSON 输出（response_format=json_object）
    """
    agent_cfg: dict[str, Any] = (config or {}).get("agent", {})
    base_url = agent_cfg.get("llm_base_url") or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    model = agent_cfg.get("llm_model") or os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 DEEPSEEK_API_KEY，请在 .env 中配置（参考 .env.example）")

    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(**kwargs)


def is_mock_mode() -> bool:
    """是否处于 MOCK_LLM 自检模式（CI / repo_check 用，不真调 API）。"""
    return os.getenv("MOCK_LLM", "").lower() in ("1", "true", "yes")
