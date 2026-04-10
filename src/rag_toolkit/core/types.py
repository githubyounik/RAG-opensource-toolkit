"""Shared dataclasses for cross-module communication."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Query:
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    query: Query
    documents: list[Document] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    query: Query
    answer: str
    contexts: list[Document] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    query: Query
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
