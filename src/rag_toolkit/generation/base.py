"""Base interface for generation components."""

from __future__ import annotations

from abc import abstractmethod

from rag_toolkit.core.base import Component
from rag_toolkit.core.types import GenerationResult, RetrievalResult


class Generator(Component):
    """Generates an answer from retrieved context."""

    @abstractmethod
    def generate(self, result: RetrievalResult) -> GenerationResult:
        """Generate a final answer from retrieval output."""

