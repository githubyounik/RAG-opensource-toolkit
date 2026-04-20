"""Factory helpers for embedding providers."""

from __future__ import annotations

from typing import Any

from rag_toolkit.embeddings.base import TextEmbedder
from rag_toolkit.embeddings.local_embedder import LocalEmbedder
from rag_toolkit.embeddings.openrouter_embedder import OpenRouterEmbedder


def create_embedder_from_config(
    embedding_config: dict[str, Any],
    *,
    openrouter_api_key: str | None = None,
) -> TextEmbedder:
    """Create an embedding client from config.

    Supported providers:
    - ``openrouter``
    - ``local``
    """

    provider = str(embedding_config["provider"]).lower()
    model = str(embedding_config["model"])

    if provider == "openrouter":
        if not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when embeddings.provider is 'openrouter'."
            )
        return OpenRouterEmbedder(
            api_key=openrouter_api_key,
            model=model,
        )

    if provider == "local":
        return LocalEmbedder(
            model=model,
            max_length=int(embedding_config.get("max_length", 512)),
            batch_size=int(embedding_config.get("batch_size", 32)),
            device=str(embedding_config.get("device", "auto")),
            pooling_method=str(embedding_config.get("pooling_method", "mean")),
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")
