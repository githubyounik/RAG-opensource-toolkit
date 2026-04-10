"""Base interface for evaluation components."""

from __future__ import annotations

from abc import abstractmethod

from rag_toolkit.core.base import Component
from rag_toolkit.core.types import EvaluationResult, GenerationResult


class Evaluator(Component):
    """Evaluates the generated answer or pipeline output."""

    @abstractmethod
    def evaluate(self, result: GenerationResult) -> EvaluationResult:
        """Evaluate generation quality with one or more metrics."""

