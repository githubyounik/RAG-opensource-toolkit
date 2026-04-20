"""Factory helpers for shared LLM provider clients."""

from __future__ import annotations

from typing import Any

from rag_toolkit.llm.base import ChatLLMClient
from rag_toolkit.llm.local_client import LocalChatClient
from rag_toolkit.llm.openrouter_client import OpenRouterChatClient
from rag_toolkit.llm.zhipu_client import ZhipuChatClient


def create_chat_llm_client(
    llm_config: dict[str, Any],
    *,
    openrouter_api_key: str | None = None,
    zhipu_api_key: str | None = None,
) -> ChatLLMClient:
    """Create a chat client from provider config.

    Supported providers:
    - ``openrouter``
    - ``zhipu``
    - ``local``
    """

    provider = str(llm_config["provider"]).lower()
    model = str(llm_config["model"])

    if provider == "openrouter":
        if not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when provider is 'openrouter'."
            )
        return OpenRouterChatClient(
            api_key=openrouter_api_key,
            model=model,
            max_retries=int(llm_config.get("max_retries", 2)),
            retry_delay_seconds=float(llm_config.get("retry_delay_seconds", 2.0)),
        )

    if provider == "zhipu":
        if not zhipu_api_key:
            raise ValueError("ZHIPU_API_KEY is required when provider is 'zhipu'.")
        return ZhipuChatClient(
            api_key=zhipu_api_key,
            model=model,
        )

    if provider == "local":
        return LocalChatClient(
            model=model,
            device=str(llm_config.get("device", "auto")),
            max_length=int(llm_config.get("max_length", 2048)),
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
