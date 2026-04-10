"""Very small query preprocessing for the basic RAG flow."""

from __future__ import annotations

from rag_toolkit.core.types import Query
from rag_toolkit.pre_retrieval.base import PreRetriever


class SimplePreRetriever(PreRetriever):
    """Normalize the input query before retrieval."""

    def process(self, query: Query) -> Query:
        # Keep preprocessing minimal: trim whitespace and lowercase the query
        # so the token matching in the retriever is more stable.
        normalized_text = " ".join(query.text.strip().lower().split())
        return Query(text=normalized_text, metadata=dict(query.metadata))

