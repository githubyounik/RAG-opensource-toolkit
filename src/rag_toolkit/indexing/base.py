"""Base interfaces for indexing components."""

from __future__ import annotations

from abc import abstractmethod

from rag_toolkit.core.base import Component
from rag_toolkit.core.types import Document, ParsedFile


class FileLoader(Component):
    """Load a file from disk and return a ParsedFile.

    Subclass this to support a new file format (PDF, Word, HTML, etc.).
    The output is always a :class:`~rag_toolkit.core.types.ParsedFile` so the
    rest of the pipeline is format-agnostic.
    """

    @abstractmethod
    def load(self, path: str) -> ParsedFile:
        """Load the file at *path* and return its parsed content."""


class TextProcessor(Component):
    """Turn a ParsedFile into a list of Document chunks.

    Responsibilities: whitespace normalisation, cleaning, and splitting the
    text into chunks that downstream retrieval stages can operate on.
    """

    @abstractmethod
    def process(self, parsed_file: ParsedFile) -> list[Document]:
        """Clean and chunk *parsed_file* into indexable Documents."""


class IndexBuilder(Component):
    """Builds or updates a retrieval index from documents."""

    @abstractmethod
    def build(self, documents: list[Document]) -> None:
        """Build the index from a collection of documents."""
