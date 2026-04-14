"""LLM-based reranking for post-retrieval refinement."""

from __future__ import annotations

from rag_toolkit.core.types import Document, RetrievalResult
from rag_toolkit.llm import ChatLLMClient
from rag_toolkit.post_retrieval.base import PostRetriever

_SYSTEM_PROMPT = """
You are reranking retrieved documents for a Retrieval-Augmented Generation system.

Given a user query and one retrieved document, assign a relevance score from 0 to 10.

Requirements:
1. Focus on how useful the document is for answering the query.
2. Consider semantic relevance, not just keyword overlap.
3. Return only the numeric score.
4. Use 0 for completely irrelevant and 10 for highly relevant.
""".strip()


class LLMReranker(PostRetriever):
    """Re-score retrieved chunks with an LLM and sort by the new score."""

    def __init__(
        self,
        client: ChatLLMClient,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = 32,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k

    def _build_user_prompt(self, *, query_text: str, document_text: str) -> str:
        """Build the prompt used to score one query-document pair."""

        return (
            f"Query:\n{query_text}\n\n"
            f"Document:\n{document_text}\n\n"
            "Relevance score (0-10):"
        )

    def _parse_score(self, raw_text: str) -> float:
        """Parse the rerank score from model output.

        The prompt asks for a bare number, but we still defensively parse and
        clamp the value in case the model returns extra text.
        """

        stripped = raw_text.strip()
        token = stripped.split()[0] if stripped else "0"
        try:
            score = float(token)
        except ValueError:
            score = 0.0
        return max(0.0, min(10.0, score))

    def _score_document(self, *, query_text: str, document: Document) -> float:
        """Use the LLM to score one retrieved document."""

        response = self.client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(
                query_text=query_text,
                document_text=document.text,
            ),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self._parse_score(response.text)

    def process(self, result: RetrievalResult) -> RetrievalResult:
        """Rerank retrieved documents with LLM scores."""

        scored_documents: list[tuple[float, int, Document]] = []

        for original_rank, document in enumerate(result.documents):
            rerank_score = self._score_document(
                query_text=result.query.text,
                document=document,
            )
            scored_documents.append(
                (
                    rerank_score,
                    original_rank,
                    Document(
                        doc_id=document.doc_id,
                        text=document.text,
                        metadata={
                            **document.metadata,
                            "original_rank": original_rank,
                            "rerank_score": rerank_score,
                            "post_retrieval_strategy": "rerank_llm",
                        },
                    ),
                )
            )

        scored_documents.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        reranked_documents = [document for _, _, document in scored_documents]
        if self.top_k is not None:
            reranked_documents = reranked_documents[: self.top_k]

        return RetrievalResult(
            query=result.query,
            documents=reranked_documents,
            metadata={
                **result.metadata,
                "post_retrieval_strategy": "rerank_llm",
                "original_retrieved_count": len(result.documents),
                "reranked_count": len(reranked_documents),
            },
        )
