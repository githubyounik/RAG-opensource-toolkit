"""Minimal evaluation for the basic RAG flow."""

from __future__ import annotations

from rag_toolkit.core.types import EvaluationResult, GenerationResult
from rag_toolkit.evaluation.base import Evaluator


class SimpleEvaluator(Evaluator):
    """Return a few basic flow-level metrics.

    This stays intentionally lightweight and only reports whether
    retrieval produced any context and how many chunks were used.
    """

    def evaluate(self, result: GenerationResult) -> EvaluationResult:
        metrics = {
            "context_count": float(len(result.contexts)),
            "has_context": 1.0 if result.contexts else 0.0,
        }
        return EvaluationResult(
            query=result.query,
            metrics=metrics,
            metadata=dict(result.metadata),
        )

