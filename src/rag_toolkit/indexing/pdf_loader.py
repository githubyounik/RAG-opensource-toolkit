"""PDF file loader."""

from __future__ import annotations

from pathlib import Path

from rag_toolkit.core.types import ParsedFile
from rag_toolkit.indexing.base import FileLoader


class PDFLoader(FileLoader):
    """Load a PDF file and return its text as a :class:`ParsedFile`.

    Each page of the PDF becomes one entry in ``ParsedFile.pages``, preserving
    page boundaries for downstream use (e.g. page-aware metadata).
    """

    def load(self, path: str) -> ParsedFile:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "pypdf is required for PDF loading. "
                "Install project dependencies first."
            ) from exc

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return ParsedFile(
            source=Path(path).name,
            pages=pages,
            metadata={"path": str(path), "page_count": len(pages)},
        )
