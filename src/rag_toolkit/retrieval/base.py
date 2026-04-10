"""Base interface for retrieval components."""

from __future__ import annotations

from abc import abstractmethod

from rag_toolkit.core.base import Component
from rag_toolkit.core.types import Query, RetrievalResult


class Retriever(Component):
    """Retrieves relevant documents for a query."""

    @abstractmethod
    def retrieve(self, query: Query) -> RetrievalResult:
        """Return candidate documents for a query."""

