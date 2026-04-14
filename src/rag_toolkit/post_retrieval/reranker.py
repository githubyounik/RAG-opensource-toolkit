"""Dedicated reranking component backed by OpenRouter's rerank API."""

from __future__ import annotations

import time
from collections.abc import Callable

import httpx

from rag_toolkit.core.types import Document, RetrievalResult
from rag_toolkit.post_retrieval.base import PostRetriever


RequestFunc = Callable[..., httpx.Response]


class CohereReranker(PostRetriever):
    """Rerank retrieved chunks with OpenRouter's dedicated rerank endpoint.

    The selected rerank model can still be a Cohere rerank model such as
    ``cohere/rerank-v3.5``, but the HTTP request is sent through OpenRouter's
    ``/rerank`` endpoint so it can reuse the existing OpenRouter API key.
    """

    API_URL = "https://openrouter.ai/api/v1/rerank"
    DEFAULT_MODEL = "cohere/rerank-v3.5"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        top_k: int | None = None,
        max_tokens_per_doc: int | None = None,
        max_retries: int = 2,
        retry_delay_seconds: float = 2.0,
        request_func: RequestFunc | None = None,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self.model = model
        self.top_k = top_k
        self.max_tokens_per_doc = max_tokens_per_doc
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._request_func = request_func or httpx.post

    def _should_retry_status(self, status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code < 600

    def _build_payload(
        self,
        *,
        query_text: str,
        documents: list[Document],
    ) -> dict[str, object]:
        """Build the rerank API payload for one retrieval result."""

        payload: dict[str, object] = {
            "model": self.model,
            "query": query_text,
            "documents": [document.text for document in documents],
        }
        if self.top_k is not None:
            payload["top_n"] = min(self.top_k, len(documents))
        if self.max_tokens_per_doc is not None:
            payload["max_tokens_per_doc"] = self.max_tokens_per_doc
        return payload

    def _request_rerank(
        self,
        *,
        query_text: str,
        documents: list[Document],
    ) -> list[dict[str, object]]:
        """Call the rerank endpoint and return raw result items."""

        payload = self._build_payload(query_text=query_text, documents=documents)
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._request_func(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "",
                        "X-OpenRouter-Title": "RAG Toolkit",
                    },
                    json=payload,
                    timeout=60,
                )

                if response.is_error:
                    status_code = response.status_code
                    response_preview = response.text[:500]

                    if attempt < self.max_retries and self._should_retry_status(status_code):
                        time.sleep(self.retry_delay_seconds)
                        continue

                    raise RuntimeError(
                        "OpenRouter rerank request failed "
                        f"(status={status_code}, model={self.model}, attempt={attempt + 1}/"
                        f"{self.max_retries + 1}). Response: {response_preview}"
                    )

                response_json = response.json()
                results = response_json.get("results")
                if not isinstance(results, list):
                    raise RuntimeError(
                        "OpenRouter rerank response did not contain a valid results list "
                        f"(model={self.model})."
                    )
                return results

            except httpx.RequestError as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)
                    continue

                raise RuntimeError(
                    "OpenRouter rerank request failed after retries "
                    f"(model={self.model}, attempts={self.max_retries + 1}). "
                    f"Original error: {exc}"
                ) from exc

        raise RuntimeError(
            "OpenRouter rerank request failed without a successful response."
        ) from last_exception

    def process(self, result: RetrievalResult) -> RetrievalResult:
        """Rerank retrieved documents by dedicated relevance scores."""

        if not result.documents:
            return result

        raw_results = self._request_rerank(
            query_text=result.query.text,
            documents=result.documents,
        )

        reranked_documents: list[Document] = []
        for reranked_rank, item in enumerate(raw_results):
            original_index = int(item.get("index", -1))
            if original_index < 0 or original_index >= len(result.documents):
                continue

            original_document = result.documents[original_index]
            rerank_score = float(item.get("relevance_score", 0.0))
            reranked_documents.append(
                Document(
                    doc_id=original_document.doc_id,
                    text=original_document.text,
                    metadata={
                        **original_document.metadata,
                        "original_rank": original_index,
                        "reranked_rank": reranked_rank,
                        "rerank_score": rerank_score,
                        "post_retrieval_strategy": "rerank",
                        "rerank_provider": "openrouter",
                        "rerank_model": self.model,
                    },
                )
            )

        return RetrievalResult(
            query=result.query,
            documents=reranked_documents,
            metadata={
                **result.metadata,
                "post_retrieval_strategy": "rerank",
                "rerank_provider": "openrouter",
                "rerank_model": self.model,
                "original_retrieved_count": len(result.documents),
                "reranked_count": len(reranked_documents),
            },
        )
