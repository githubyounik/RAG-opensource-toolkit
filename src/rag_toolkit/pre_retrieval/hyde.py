"""HyDE pre-retrieval component.

HyDE stands for Hypothetical Document Embedding. Instead of embedding the
original short query directly, we first ask an LLM to generate a hypothetical
document that would answer the question well. That synthetic document is then
used as the retrieval text, which can better match the style and density of the
indexed chunks in vector space.
"""

from __future__ import annotations

from rag_toolkit.core.types import Query
from rag_toolkit.pre_retrieval.base import PreRetriever
from rag_toolkit.pre_retrieval.prompts import HYDE_SYSTEM_PROMPT_TEMPLATE as _SYSTEM_PROMPT_TEMPLATE
from rag_toolkit.pre_retrieval.query_transformer import QueryTransformer


class HyDEPreRetriever(PreRetriever):
    """Generate a hypothetical answer document and use it as retrieval text.

    This component keeps the current pipeline structure unchanged. It simply
    returns a new ``Query`` whose ``text`` is the hypothetical document, so the
    existing embedding retriever can continue to work without any extra changes.
    """

    def __init__(
        self,
        transformer: QueryTransformer,
        *,
        target_char_length: int | None = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.target_char_length = target_char_length

    def _build_system_prompt(self) -> str:
        """Build the HyDE instruction prompt.

        When a target length is provided, we add it as a gentle hint so the
        generated hypothetical document is closer to the size of indexed chunks.
        """

        if self.target_char_length is None:
            length_instruction = ""
        else:
            length_instruction = (
                f"6. Aim for about {self.target_char_length} characters."
            )

        return _SYSTEM_PROMPT_TEMPLATE.format(length_instruction=length_instruction)

    def process(self, query: Query) -> Query:
        """Turn the original query into a hypothetical retrieval document."""

        hypothetical_document = self.transformer.transform(
            system_prompt=self._build_system_prompt(),
            user_prompt=query.text,
        )
        return Query(
            text=hypothetical_document,
            metadata={
                **query.metadata,
                "original_query": query.text,
                "pre_retrieval_strategy": "hyde",
                "hyde_hypothetical_document": hypothetical_document,
            },
        )
