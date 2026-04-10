"""Base interface for pre-retrieval components."""

from __future__ import annotations

from abc import abstractmethod

from rag_toolkit.core.base import Component
from rag_toolkit.core.types import Query


class PreRetriever(Component):
    """Transforms or enriches a query before retrieval."""

    @abstractmethod
    def process(self, query: Query) -> Query:
        """Process a query before it reaches the retriever."""

