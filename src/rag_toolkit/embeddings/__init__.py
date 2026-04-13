"""Embedding client, vector store, and indexer."""

from rag_toolkit.embeddings.base import TextEmbedder, VectorIndexBuilder
from rag_toolkit.embeddings.factory import create_embedder_from_config
from rag_toolkit.embeddings.indexer import EmbeddingIndexer
from rag_toolkit.embeddings.openrouter_embedder import OpenRouterEmbedder
from rag_toolkit.embeddings.vector_index import VectorIndex

__all__ = [
    "EmbeddingIndexer",
    "OpenRouterEmbedder",
    "TextEmbedder",
    "VectorIndex",
    "VectorIndexBuilder",
    "create_embedder_from_config",
]
