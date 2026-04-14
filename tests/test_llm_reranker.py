from rag_toolkit.core.types import Document, Query, RetrievalResult
from rag_toolkit.post_retrieval.llm_reranker import LLMReranker


class DummyClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
    ):
        class Response:
            def __init__(self, text: str) -> None:
                self.text = text

        return Response(self.outputs.pop(0))


def test_llm_reranker_reorders_documents_by_llm_score() -> None:
    reranker = LLMReranker(
        DummyClient(["3", "9", "1"]),
        temperature=0.0,
        max_tokens=32,
        top_k=2,
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
    assert reranked.documents[0].metadata["rerank_score"] == 9.0
    assert reranked.documents[1].metadata["rerank_score"] == 3.0
    assert reranked.metadata["post_retrieval_strategy"] == "rerank_llm"
