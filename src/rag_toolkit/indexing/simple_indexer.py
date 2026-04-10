"""Basic PDF indexing implementation for the simple RAG flow."""

from __future__ import annotations

from pathlib import Path

from rag_toolkit.core.types import Document
from rag_toolkit.indexing.base import IndexBuilder
from rag_toolkit.indexing.simple_chunker import SimpleTextChunker
from rag_toolkit.indexing.simple_index import InMemoryIndex


class SimplePDFIndexer(IndexBuilder):
    """Load a PDF, split its text, and store the chunks in memory."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        super().__init__()
        self.chunker = SimpleTextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.index = InMemoryIndex()

    def load_pdf_text(self, path: str) -> str:
        """Read and join all pages from a PDF file.

        The notebook uses a PDF loader. Here we keep only the smallest
        implementation needed for the basic flow.
        """

        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "pypdf is required for PDF indexing. Install project dependencies first."
            ) from exc

        pdf_path = Path(path)
        reader = PdfReader(str(pdf_path))
        page_texts: list[str] = []
        for page in reader.pages:
            page_texts.append(page.extract_text() or "")

        return "\n".join(page_texts)

    def build(self, documents: list[Document]) -> None:
        """Store chunked documents in the in-memory index."""

        self.index = InMemoryIndex(documents=list(documents))

    def index_pdf(self, path: str) -> InMemoryIndex:
        """Full indexing flow: load PDF, chunk it, and build the index."""

        text = self.load_pdf_text(path)
        documents = self.chunker.split_text(text=text, source=Path(path).name)
        self.build(documents)
        return self.index

