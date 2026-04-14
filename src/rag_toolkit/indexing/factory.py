"""Factory helpers for indexing-stage processors."""

from __future__ import annotations

from typing import Any

from rag_toolkit.indexing.base import TextProcessor
from rag_toolkit.indexing.document_processor import DocumentProcessor
from rag_toolkit.indexing.proposition_processor import PropositionProcessor


def create_text_processor_from_config(
    document_processing_config: dict[str, Any],
    *,
    openrouter_api_key: str | None = None,
    force_non_overlapping_default: bool = False,
) -> TextProcessor:
    """Create a text processor from the indexing config.

    Supported strategies:
    - ``default``: regular fixed-size chunking with `DocumentProcessor`
    - ``proposition``: proposition-level chunking with `PropositionProcessor`
    """

    strategy = str(document_processing_config.get("strategy", "default")).lower()
    chunk_size = int(document_processing_config["chunk_size"])
    chunk_overlap = int(document_processing_config["chunk_overlap"])

    if force_non_overlapping_default:
        strategy = "default"
        chunk_overlap = 0

    if strategy == "default":
        return DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if strategy == "proposition":
        if not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when document_processing.strategy is "
                "'proposition'."
            )

        proposition_config = document_processing_config.get("proposition", {})
        return PropositionProcessor(
            api_key=openrouter_api_key,
            base_chunk_size=chunk_size,
            base_chunk_overlap=chunk_overlap,
            model=str(
                proposition_config.get(
                    "model",
                    PropositionProcessor.DEFAULT_MODEL,
                )
            ),
            temperature=float(proposition_config.get("temperature", 0.0)),
            max_tokens=(
                int(proposition_config["max_tokens"])
                if proposition_config.get("max_tokens") is not None
                else 512
            ),
            max_retries=int(proposition_config.get("max_retries", 2)),
            retry_delay_seconds=float(proposition_config.get("retry_delay_seconds", 2.0)),
        )

    raise ValueError(f"Unsupported document processing strategy: {strategy}")
