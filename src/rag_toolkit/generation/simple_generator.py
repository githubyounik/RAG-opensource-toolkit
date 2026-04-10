"""Minimal generation stage for the basic RAG flow."""

from __future__ import annotations

from rag_toolkit.core.types import GenerationResult, RetrievalResult
from rag_toolkit.generation.base import Generator


class SimpleGenerator(Generator):
    """Produce an answer from retrieved chunks.

    This does not call an LLM. It only formats the retrieved context into
    a readable answer so the end-to-end RAG flow stays runnable and easy to inspect.
    """

    def generate(self, result: RetrievalResult) -> GenerationResult:
        if not result.documents:
            answer = "No relevant context was found for the query."
        else:
            context_blocks = [document.text for document in result.documents]
            answer = "Based on the retrieved context:\n\n" + "\n\n".join(context_blocks)

        return GenerationResult(
            query=result.query,
            answer=answer,
            contexts=list(result.documents),
            metadata=dict(result.metadata),
        )

