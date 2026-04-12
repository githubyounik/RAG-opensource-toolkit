"""Vector index that stores document chunks alongside their embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_toolkit.core.types import Document


@dataclass(slots=True)
class VectorIndex:
    """In-memory store of documents and their embedding vectors.

    ``documents`` and ``embeddings`` are kept in sync: ``embeddings[i]`` is
    the embedding for ``documents[i]``.
    """

    documents: list[Document] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)

    def add(self, document: Document, embedding: list[float]) -> None:
        """Append a document and its embedding together."""
        self.documents.append(document)
        self.embeddings.append(embedding)

    def __len__(self) -> int:
        return len(self.documents)
