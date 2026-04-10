"""Simple in-memory index for the basic RAG flow."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_toolkit.core.types import Document


@dataclass(slots=True)
class InMemoryIndex:
    """A minimal index that stores document chunks in memory."""

    documents: list[Document] = field(default_factory=list)

