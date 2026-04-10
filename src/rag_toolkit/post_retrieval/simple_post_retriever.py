"""Minimal post-retrieval processing."""

from __future__ import annotations

from rag_toolkit.core.types import RetrievalResult
from rag_toolkit.post_retrieval.base import PostRetriever


class SimplePostRetriever(PostRetriever):
    """Keep only non-empty retrieved chunks.

    The notebook's simple flow does not perform reranking, so this stage
    intentionally stays close to a no-op.
    """

    def process(self, result: RetrievalResult) -> RetrievalResult:
        filtered_documents = [document for document in result.documents if document.text.strip()]
        return RetrievalResult(
            query=result.query,
            documents=filtered_documents,
            metadata=dict(result.metadata),
        )

