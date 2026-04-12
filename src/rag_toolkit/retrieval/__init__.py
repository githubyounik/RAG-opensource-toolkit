"""Retrieval module interfaces."""

from rag_toolkit.retrieval.base import Retriever
from rag_toolkit.retrieval.embedding_retriever import EmbeddingRetriever

__all__ = ["EmbeddingRetriever", "Retriever"]
