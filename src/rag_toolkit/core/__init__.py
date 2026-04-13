"""Shared abstractions and data structures."""

from rag_toolkit.core.base import Component
from rag_toolkit.core.config import load_pipeline_config
from rag_toolkit.core.types import Document, EvaluationResult, GenerationResult, Query, RetrievalResult

__all__ = [
    "Component",
    "Document",
    "EvaluationResult",
    "GenerationResult",
    "Query",
    "RetrievalResult",
    "load_pipeline_config",
]
