"""Step-back query generation for pre-retrieval."""

from __future__ import annotations

from rag_toolkit.core.types import Query
from rag_toolkit.pre_retrieval.base import PreRetriever
from rag_toolkit.pre_retrieval.query_transformer import QueryTransformer

_SYSTEM_PROMPT = """
You improve retrieval in a RAG system by generating a broader step-back query.

Rewrite the user's query into a more general background query that helps
retrieve supporting context related to the same topic.

Return only the step-back query text.
""".strip()


class StepBackPreRetriever(PreRetriever):
    """Generate a broader step-back query for context retrieval."""

    def __init__(self, transformer: QueryTransformer) -> None:
        super().__init__()
        self.transformer = transformer

    def process(self, query: Query) -> Query:
        step_back_query = self.transformer.transform(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=query.text,
        )
        return Query(
            text=step_back_query,
            metadata={
                **query.metadata,
                "original_query": query.text,
                "pre_retrieval_strategy": "step_back",
            },
        )
