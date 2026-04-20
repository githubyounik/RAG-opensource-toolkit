"""Step-back query generation for pre-retrieval."""

from __future__ import annotations

from rag_toolkit.core.types import Query
from rag_toolkit.pre_retrieval.base import PreRetriever
from rag_toolkit.pre_retrieval.prompts import STEP_BACK_SYSTEM_PROMPT as _SYSTEM_PROMPT
from rag_toolkit.pre_retrieval.query_transformer import QueryTransformer


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
