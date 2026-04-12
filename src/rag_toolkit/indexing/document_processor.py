"""Text preprocessing and chunking."""

from __future__ import annotations

from rag_toolkit.core.types import Document, ParsedFile
from rag_toolkit.indexing.base import TextProcessor


class DocumentProcessor(TextProcessor):
    """Clean and chunk a :class:`ParsedFile` into indexable Documents.

    Steps applied in order:
    1. Join all pages with a newline.
    2. Collapse whitespace (strips extra spaces, tabs, newlines).
    3. Split into overlapping fixed-size character chunks.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk.
    chunk_overlap:
        Number of characters shared between adjacent chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self, parsed_file: ParsedFile) -> list[Document]:
        """Clean and chunk *parsed_file* into Documents."""
        raw_text = "\n".join(parsed_file.pages)
        text = " ".join(raw_text.split())
        if not text:
            return []

        documents: list[Document] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_id = 0

        while start < len(text):
            chunk_text = text[start : start + self.chunk_size].strip()
            if chunk_text:
                documents.append(
                    Document(
                        doc_id=f"{parsed_file.source}-chunk-{chunk_id}",
                        text=chunk_text,
                        metadata={
                            "source": parsed_file.source,
                            "chunk_id": chunk_id,
                            "start": start,
                            "end": min(start + self.chunk_size, len(text)),
                        },
                    )
                )
            start += step
            chunk_id += 1

        return documents
