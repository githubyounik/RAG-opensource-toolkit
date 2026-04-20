"""Cosine-similarity retriever backed by a VectorIndex and OpenRouter embeddings."""

from __future__ import annotations

import math

from rag_toolkit.core.types import Document, Query, RetrievalResult
from rag_toolkit.embeddings.openrouter_embedder import OpenRouterEmbedder
from rag_toolkit.embeddings.vector_index import VectorIndex
from rag_toolkit.retrieval.base import Retriever


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingRetriever(Retriever):
    """Retrieve chunks by embedding the query and ranking by cosine similarity.

    Parameters
    ----------
    index:
        A :class:`~rag_toolkit.embeddings.vector_index.VectorIndex` built by
        :class:`~rag_toolkit.embeddings.pdf_indexer.EmbeddingPDFIndexer`.
    embedder:
        The same :class:`OpenRouterEmbedder` used at index time (must use the
        same model so embedding spaces match).
    top_k:
        Number of top-ranked documents to return.
    """

    def __init__(
        self,
        index: VectorIndex,
        embedder: OpenRouterEmbedder,
        top_k: int = 4,
    ) -> None:
        super().__init__()
        self.index = index
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: Query) -> RetrievalResult:
        """Embed the query and return the top-k most similar documents."""
        query_embedding = self.embedder.embed_one(query.text)

        if self.index.faiss_index is not None:
            scored = self.index.search(query_embedding, self.top_k)
        else:
            scored: list[tuple[float, Document]] = []
            for doc, doc_emb in zip(self.index.documents, self.index.embeddings):
                score = _cosine_similarity(query_embedding, doc_emb)
                scored_doc = Document(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    metadata={**doc.metadata, "score": score},
                )
                scored.append((score, scored_doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_documents = [doc for _, doc in scored[: self.top_k]]

        return RetrievalResult(
            query=query,
            documents=top_documents,
            metadata={
                "top_k": self.top_k,
                "index_size": len(self.index),
                "faiss_enabled": self.index.faiss_index is not None,
            },
        )
