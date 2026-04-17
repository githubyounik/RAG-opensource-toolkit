"""Retrieval module interfaces."""

from rag_toolkit.retrieval.base import Retriever
from rag_toolkit.retrieval.bm25_retriever import BM25Retriever
from rag_toolkit.retrieval.embedding_retriever import EmbeddingRetriever
from rag_toolkit.retrieval.factory import create_retriever_from_config
from rag_toolkit.retrieval.hybrid_retriever import HybridRetriever

__all__ = [
    "BM25Retriever",
    "EmbeddingRetriever",
    "HybridRetriever",
    "Retriever",
    "create_retriever_from_config",
]
