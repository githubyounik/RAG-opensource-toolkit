"""Simple text chunking utilities used by the basic indexer."""

from __future__ import annotations

from rag_toolkit.core.types import Document


class SimpleTextChunker:
    """Split long text into overlapping chunks.

    This keeps the implementation intentionally small and readable.
    The behavior mirrors the notebook idea of chunking a document before retrieval.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, source: str) -> list[Document]:
        """Split text into chunk documents.

        Each chunk carries simple metadata so later stages can trace
        where the text came from.
        """

        normalized_text = " ".join(text.split())
        if not normalized_text:
            return []

        documents: list[Document] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_id = 0

        while start < len(normalized_text):
            end = start + self.chunk_size
            chunk_text = normalized_text[start:end].strip()
            if chunk_text:
                documents.append(
                    Document(
                        doc_id=f"{source}-chunk-{chunk_id}",
                        text=chunk_text,
                        metadata={
                            "source": source,
                            "chunk_id": chunk_id,
                            "start": start,
                            "end": min(end, len(normalized_text)),
                        },
                    )
                )

            start += step
            chunk_id += 1

        return documents

