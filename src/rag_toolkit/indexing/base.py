"""Base interface for indexing components."""

from __future__ import annotations

from abc import abstractmethod

from rag_toolkit.core.base import Component
from rag_toolkit.core.types import Document


class IndexBuilder(Component):
    """Builds or updates a retrieval index from documents."""

    @abstractmethod
    def build(self, documents: list[Document]) -> None:
        """Build the index from a collection of documents."""
