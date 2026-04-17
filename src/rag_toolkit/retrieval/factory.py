"""Factory helpers for retrieval components."""

from __future__ import annotations

from typing import Any

from rag_toolkit.core.types import Document
from rag_toolkit.embeddings.base import TextEmbedder
from rag_toolkit.embeddings.vector_index import VectorIndex
from rag_toolkit.retrieval.base import Retriever
from rag_toolkit.retrieval.bm25_retriever import BM25Retriever
from rag_toolkit.retrieval.embedding_retriever import EmbeddingRetriever


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
    """

    retrieval_config = retrieval_config or {}
    strategy = str(retrieval_config.get("strategy", "embedding")).lower()

    if strategy == "embedding":
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

    if strategy == "bm25":
        bm25_config = retrieval_config.get("bm25", {})
        return BM25Retriever(
            documents=documents,
            top_k=int(bm25_config.get("top_k", retrieval_config.get("top_k", 4))),
            k1=float(bm25_config.get("k1", 1.5)),
            b=float(bm25_config.get("b", 0.75)),
            lowercase=bool(bm25_config.get("lowercase", True)),
        )

    raise ValueError(f"Unsupported retrieval strategy: {strategy}")
