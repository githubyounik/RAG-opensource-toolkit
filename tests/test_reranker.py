from __future__ import annotations

import httpx

from rag_toolkit.core.types import Document, Query, RetrievalResult
from rag_toolkit.post_retrieval.reranker import CohereReranker


def _fake_response(*args, **kwargs) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "results": [
                {"index": 1, "relevance_score": 0.92},
                {"index": 0, "relevance_score": 0.33},
            ]
        },
    )


def test_cohere_reranker_reorders_documents_by_relevance_score() -> None:
    reranker = CohereReranker(
        api_key="test-key",
        model="cohere/rerank-v3.5",
        top_k=2,
        request_func=_fake_response,
    )

    result = RetrievalResult(
        query=Query(text="Which chunk is most relevant?"),
        documents=[
            Document(doc_id="doc-a", text="A", metadata={}),
            Document(doc_id="doc-b", text="B", metadata={}),
            Document(doc_id="doc-c", text="C", metadata={}),
        ],
    )

    reranked = reranker.process(result)

    assert [document.doc_id for document in reranked.documents] == ["doc-b", "doc-a"]
    assert reranked.documents[0].metadata["rerank_score"] == 0.92
    assert reranked.documents[1].metadata["rerank_score"] == 0.33
    assert reranked.metadata["post_retrieval_strategy"] == "rerank"
    assert reranked.metadata["rerank_model"] == "cohere/rerank-v3.5"
    assert reranked.metadata["rerank_provider"] == "openrouter"
