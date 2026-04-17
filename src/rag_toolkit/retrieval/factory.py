"""Factory helpers for retrieval components."""

from __future__ import annotations

from typing import Any

from rag_toolkit.core.types import Document
from rag_toolkit.embeddings.base import TextEmbedder
from rag_toolkit.embeddings.vector_index import VectorIndex
from rag_toolkit.retrieval.base import Retriever
from rag_toolkit.retrieval.bm25_retriever import BM25Retriever
from rag_toolkit.retrieval.embedding_retriever import EmbeddingRetriever
from rag_toolkit.retrieval.hybrid_retriever import HybridRetriever


def _create_embedding_retriever(
    retrieval_config: dict[str, Any],
    *,
    index: VectorIndex | None,
    embedder: TextEmbedder | None,
) -> EmbeddingRetriever:
    embedding_config = retrieval_config.get("embedding", {})
    if index is None:
        raise ValueError("A VectorIndex is required when retrieval.strategy is 'embedding'.")
    if embedder is None:
        raise ValueError("An embedder is required when retrieval.strategy is 'embedding'.")
    return EmbeddingRetriever(
        index=index,
        embedder=embedder,
        top_k=int(embedding_config.get("top_k", retrieval_config.get("top_k", 4))),
    )


def _create_bm25_retriever(
    retrieval_config: dict[str, Any],
    *,
    documents: list[Document],
) -> BM25Retriever:
    bm25_config = retrieval_config.get("bm25", {})
    return BM25Retriever(
        documents=documents,
        top_k=int(bm25_config.get("top_k", retrieval_config.get("top_k", 4))),
        k1=float(bm25_config.get("k1", 1.5)),
        b=float(bm25_config.get("b", 0.75)),
        lowercase=bool(bm25_config.get("lowercase", True)),
    )


def create_retriever_from_config(
    retrieval_config: dict[str, Any] | None,
    *,
    documents: list[Document],
    index: VectorIndex | None = None,
    embedder: TextEmbedder | None = None,
) -> Retriever:
    """Create a Retriever from config.

    Supported strategies:
    - ``embedding``: dense retrieval over a pre-built ``VectorIndex``
    - ``bm25``: sparse BM25 retrieval directly over ``Document`` chunks
    - ``hybrid``: fuse embedding and BM25 rankings with Reciprocal Rank Fusion
    """

    retrieval_config = retrieval_config or {}
    strategy = str(retrieval_config.get("strategy", "embedding")).lower()

    if strategy == "embedding":
        return _create_embedding_retriever(
            retrieval_config,
            index=index,
            embedder=embedder,
        )

    if strategy == "bm25":
        return _create_bm25_retriever(
            retrieval_config,
            documents=documents,
        )

    if strategy == "hybrid":
        hybrid_config = retrieval_config.get("hybrid", {})
        embedding_retriever = _create_embedding_retriever(
            retrieval_config,
            index=index,
            embedder=embedder,
        )
        bm25_retriever = _create_bm25_retriever(
            retrieval_config,
            documents=documents,
        )
        return HybridRetriever(
            embedding_retriever=embedding_retriever,
            bm25_retriever=bm25_retriever,
            documents=documents,
            top_k=int(hybrid_config.get("top_k", retrieval_config.get("top_k", 4))),
            rrf_k=int(hybrid_config.get("rrf_k", 60)),
        )

    raise ValueError(f"Unsupported retrieval strategy: {strategy}")
