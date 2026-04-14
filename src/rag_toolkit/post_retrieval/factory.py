"""Factory helpers for post-retrieval components."""

from __future__ import annotations

from typing import Any

from rag_toolkit.core.types import Document
from rag_toolkit.llm import create_chat_llm_client
from rag_toolkit.post_retrieval.base import PostRetriever
from rag_toolkit.post_retrieval.contextual_compressor import ContextualCompressor
from rag_toolkit.post_retrieval.relevant_segment_extractor import RelevantSegmentExtractor


def create_post_retriever_from_config(
    post_retrieval_config: dict[str, Any] | None,
    *,
    documents: list[Document],
    openrouter_api_key: str | None = None,
    zhipu_api_key: str | None = None,
) -> PostRetriever | None:
    """Create a post-retrieval component from config."""

    if not post_retrieval_config or not bool(post_retrieval_config.get("enabled", False)):
        return None

    strategy = str(
        post_retrieval_config.get("strategy", "relevant_segment_extraction")
    ).lower()

    if strategy == "relevant_segment_extraction":
        return RelevantSegmentExtractor(
            documents,
            irrelevant_chunk_penalty=float(
                post_retrieval_config.get("irrelevant_chunk_penalty", 0.2)
            ),
            rank_decay=float(post_retrieval_config.get("rank_decay", 0.08)),
            max_segment_length=int(post_retrieval_config.get("max_segment_length", 6)),
            overall_max_length=int(post_retrieval_config.get("overall_max_length", 12)),
            minimum_segment_value=float(
                post_retrieval_config.get("minimum_segment_value", 0.15)
            ),
        )

    if strategy == "contextual_compression":
        llm_client = create_chat_llm_client(
            post_retrieval_config,
            openrouter_api_key=openrouter_api_key,
            zhipu_api_key=zhipu_api_key,
        )
        return ContextualCompressor(
            llm_client,
            temperature=float(post_retrieval_config.get("temperature", 0.0)),
            max_tokens=(
                int(post_retrieval_config["max_tokens"])
                if post_retrieval_config.get("max_tokens") is not None
                else 256
            ),
        )

    raise ValueError(f"Unsupported post-retrieval strategy: {strategy}")
