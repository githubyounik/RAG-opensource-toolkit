"""Base interface for post-retrieval components."""

from __future__ import annotations

from abc import abstractmethod

from rag_toolkit.core.base import Component
from rag_toolkit.core.types import RetrievalResult


class PostRetriever(Component):
    """Refines retrieval outputs before generation."""

    @abstractmethod
    def process(self, result: RetrievalResult) -> RetrievalResult:
        """Refine retrieved documents for downstream generation."""

