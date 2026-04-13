"""Query rewrite for pre-retrieval."""

from __future__ import annotations

from rag_toolkit.core.types import Query
from rag_toolkit.pre_retrieval.base import PreRetriever
from rag_toolkit.pre_retrieval.query_transformer import QueryTransformer

_SYSTEM_PROMPT = """
You improve search queries for retrieval in a RAG system.

Rewrite the user's query so it is:
1. More specific
2. More retrieval-friendly
3. Focused on the same original intent
4. Written as a single search query

Return only the rewritten query text.
""".strip()


class QueryRewritePreRetriever(PreRetriever):
    """Rewrite a user query to improve retrieval quality."""

    def __init__(self, transformer: QueryTransformer) -> None:
        super().__init__()
        self.transformer = transformer

    def process(self, query: Query) -> Query:
        rewritten_query = self.transformer.transform(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=query.text,
        )
        return Query(
            text=rewritten_query,
            metadata={
                **query.metadata,
                "original_query": query.text,
                "pre_retrieval_strategy": "rewrite",
            },
        )
