"""Contextual compression post-retrieval component.

This component compresses each retrieved document down to only the portions
that are relevant to the current query. It is inspired by the contextual
compression notebook, but adapted to the current modular toolkit structure.
"""

from __future__ import annotations

from rag_toolkit.core.types import Document, RetrievalResult
from rag_toolkit.llm import ChatLLMClient
from rag_toolkit.post_retrieval.base import PostRetriever

_SYSTEM_PROMPT = """
You are compressing retrieved context for a Retrieval-Augmented Generation system.

Given a user query and a retrieved document chunk:
1. Keep only the information that is directly useful for answering the query.
2. Remove irrelevant sentences, repetition, and tangential details.
3. Preserve important facts, quantities, dates, names, and qualifiers.
4. Return a concise extract or summary grounded only in the provided document.
5. If the document is not useful for the query, return exactly: NONE

Return only the compressed context text or NONE.
""".strip()


class ContextualCompressor(PostRetriever):
    """Compress retrieved chunks into query-focused context."""

    def __init__(
        self,
        client: ChatLLMClient,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = 256,
    ) -> None:
        super().__init__()
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_user_prompt(self, *, query_text: str, document_text: str) -> str:
        """Build the compression prompt for one retrieved document."""

        return (
            f"Query:\n{query_text}\n\n"
            f"Retrieved document:\n{document_text}\n\n"
            "Compressed relevant context:"
        )

    def _compress_document(self, *, query_text: str, document: Document) -> Document | None:
        """Compress a single retrieved document.

        If the model decides the document is not useful, this method returns
        ``None`` so the chunk is removed from downstream generation context.
        """

        response = self.client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(
                query_text=query_text,
                document_text=document.text,
            ),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        compressed_text = response.text.strip()

        if not compressed_text or compressed_text.upper() == "NONE":
            return None

        return Document(
            doc_id=f"{document.doc_id}-compressed",
            text=compressed_text,
            metadata={
                **document.metadata,
                "original_doc_id": document.doc_id,
                "original_text_length": len(document.text),
                "compressed_text_length": len(compressed_text),
                "post_retrieval_strategy": "contextual_compression",
            },
        )

    def process(self, result: RetrievalResult) -> RetrievalResult:
        """Compress all retrieved documents for the current query."""

        compressed_documents: list[Document] = []
        for document in result.documents:
            compressed = self._compress_document(
                query_text=result.query.text,
                document=document,
            )
            if compressed is not None:
                compressed_documents.append(compressed)

        # If compression yields nothing useful, keep the original retrieval
        # result so downstream generation still has context.
        if not compressed_documents:
            return result

        return RetrievalResult(
            query=result.query,
            documents=compressed_documents,
            metadata={
                **result.metadata,
                "post_retrieval_strategy": "contextual_compression",
                "original_retrieved_count": len(result.documents),
                "compressed_document_count": len(compressed_documents),
            },
        )
