"""Factory helpers for generation providers."""

from __future__ import annotations

from typing import Any

from rag_toolkit.generation.base import Generator
from rag_toolkit.generation.openrouter_generator import OpenRouterGenerator
from rag_toolkit.generation.zhipu_generator import ZhipuGenerator


def create_generator_from_config(
    generation_config: dict[str, Any],
    *,
    openrouter_api_key: str | None = None,
    zhipu_api_key: str | None = None,
) -> Generator:
    """Create a configured Generator instance from the YAML config.

    Supported providers:
    - ``zhipu``
    - ``openrouter``
    """

    provider = str(generation_config["provider"]).lower()
    model = str(generation_config["model"])
    temperature = float(generation_config.get("temperature", 0.6))
    max_tokens = generation_config.get("max_tokens")

    if provider == "zhipu":
        if not zhipu_api_key:
            raise ValueError("ZHIPU_API_KEY is required when generation.provider is 'zhipu'.")
        return ZhipuGenerator(
            api_key=zhipu_api_key,
            model=model,
            temperature=temperature,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
        )

    if provider == "openrouter":
        if not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when generation.provider is 'openrouter'."
            )
        return OpenRouterGenerator(
            api_key=openrouter_api_key,
            model=model,
            temperature=temperature,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
        )

    raise ValueError(f"Unsupported generation provider: {provider}")
