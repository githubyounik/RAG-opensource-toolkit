"""Factory helpers for post-retrieval components."""

from __future__ import annotations

from typing import Any

from rag_toolkit.core.types import Document
from rag_toolkit.llm import create_chat_llm_client
from rag_toolkit.post_retrieval.base import PostRetriever
from rag_toolkit.post_retrieval.contextual_compressor import ContextualCompressor
from rag_toolkit.post_retrieval.llm_reranker import LLMReranker
from rag_toolkit.post_retrieval.relevant_segment_extractor import RelevantSegmentExtractor


def _get_strategy_config(
    post_retrieval_config: dict[str, Any],
    strategy: str,
) -> dict[str, Any]:
    """Return the nested config section for the selected strategy."""

    raw_config = post_retrieval_config.get(strategy, {})
    if not isinstance(raw_config, dict):
        raise ValueError(
            f"post_retrieval.{strategy} must be a YAML mapping of strategy parameters."
        )
    return raw_config


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
    strategy_config = _get_strategy_config(post_retrieval_config, strategy)

    if strategy == "relevant_segment_extraction":
        return RelevantSegmentExtractor(
            documents,
            irrelevant_chunk_penalty=float(strategy_config.get("irrelevant_chunk_penalty", 0.2)),
            rank_decay=float(strategy_config.get("rank_decay", 0.08)),
            max_segment_length=int(strategy_config.get("max_segment_length", 6)),
            overall_max_length=int(strategy_config.get("overall_max_length", 12)),
            minimum_segment_value=float(
                strategy_config.get("minimum_segment_value", 0.15)
            ),
        )

    if strategy == "contextual_compression":
        llm_client = create_chat_llm_client(
            strategy_config,
            openrouter_api_key=openrouter_api_key,
            zhipu_api_key=zhipu_api_key,
        )
        return ContextualCompressor(
            llm_client,
            temperature=float(strategy_config.get("temperature", 0.0)),
            max_tokens=(
                int(strategy_config["max_tokens"])
                if strategy_config.get("max_tokens") is not None
                else 256
            ),
        )

    if strategy == "rerank_llm":
        llm_client = create_chat_llm_client(
            strategy_config,
            openrouter_api_key=openrouter_api_key,
            zhipu_api_key=zhipu_api_key,
        )
        return LLMReranker(
            llm_client,
            temperature=float(strategy_config.get("temperature", 0.0)),
            max_tokens=(
                int(strategy_config["max_tokens"])
                if strategy_config.get("max_tokens") is not None
                else 32
            ),
            top_k=(
                int(strategy_config["top_k"])
                if strategy_config.get("top_k") is not None
                else None
            ),
        )

    raise ValueError(f"Unsupported post-retrieval strategy: {strategy}")
