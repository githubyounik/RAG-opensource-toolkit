"""Base interfaces for embedding components."""

from __future__ import annotations

from abc import abstractmethod

from rag_toolkit.core.base import Component
from rag_toolkit.core.types import Document
from rag_toolkit.embeddings.vector_index import VectorIndex


class TextEmbedder(Component):
    """Embed text into dense vectors.

    Subclass this to add a new embedding backend (OpenRouter, OpenAI,
    local sentence-transformers, etc.).
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for a single text."""
        return self.embed([text])[0]


class VectorIndexBuilder(Component):
    """Build a VectorIndex from a list of Documents.

    Subclass this to change how documents are embedded and stored
    (e.g. different batching strategies, async calls, etc.).
    """

    @abstractmethod
    def build(self, documents: list[Document]) -> VectorIndex:
        """Embed *documents* and return a populated VectorIndex."""
