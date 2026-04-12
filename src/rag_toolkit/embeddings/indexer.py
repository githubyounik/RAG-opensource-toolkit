"""Embedding indexer: converts Documents into a searchable VectorIndex."""

from __future__ import annotations

from rag_toolkit.core.types import Document
from rag_toolkit.embeddings.base import TextEmbedder, VectorIndexBuilder
from rag_toolkit.embeddings.vector_index import VectorIndex


class EmbeddingIndexer(VectorIndexBuilder):
    """Embed a list of Documents and store them in a VectorIndex.

    Parameters
    ----------
    embedder:
        Any :class:`TextEmbedder` implementation.
    batch_size:
        Number of documents sent to the embedding API per request.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.batch_size = batch_size

    def build(self, documents: list[Document]) -> VectorIndex:
        """Embed *documents* and return a populated VectorIndex."""
        index = VectorIndex()
        for batch_start in range(0, len(documents), self.batch_size):
            batch = documents[batch_start : batch_start + self.batch_size]
            embeddings = self.embedder.embed([doc.text for doc in batch])
            for doc, emb in zip(batch, embeddings):
                index.add(doc, emb)
        return index
