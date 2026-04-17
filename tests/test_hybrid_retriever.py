from rag_toolkit.core.types import Document, Query
from rag_toolkit.retrieval import HybridRetriever
from rag_toolkit.retrieval.base import Retriever


class _FixedRetriever(Retriever):
    def __init__(self, documents: list[Document]) -> None:
        super().__init__()
        self._documents = documents

    def retrieve(self, query: Query):  # type: ignore[override]
        from rag_toolkit.core.types import RetrievalResult

        return RetrievalResult(query=query, documents=self._documents, metadata={})


def test_hybrid_retriever_fuses_rankings_with_rrf() -> None:
    doc_a = Document(doc_id="a", text="doc a", metadata={})
    doc_b = Document(doc_id="b", text="doc b", metadata={})
    doc_c = Document(doc_id="c", text="doc c", metadata={})

    embedding_docs = [
        Document(doc_id="a", text="doc a", metadata={"score": 0.9}),
        Document(doc_id="b", text="doc b", metadata={"score": 0.8}),
    ]
    bm25_docs = [
        Document(doc_id="b", text="doc b", metadata={"score": 9.0}),
        Document(doc_id="c", text="doc c", metadata={"score": 8.0}),
    ]

    retriever = HybridRetriever(
        embedding_retriever=_FixedRetriever(embedding_docs),
        bm25_retriever=_FixedRetriever(bm25_docs),
        documents=[doc_a, doc_b, doc_c],
        top_k=3,
        rrf_k=60,
    )

    result = retriever.retrieve(Query(text="test query"))

    assert [document.doc_id for document in result.documents] == ["b", "a", "c"]
    assert result.metadata["retrieval_strategy"] == "hybrid_rrf"
    assert result.documents[0].metadata["embedding_rank"] == 2
    assert result.documents[0].metadata["bm25_rank"] == 1


def test_hybrid_retriever_respects_top_k() -> None:
    doc_a = Document(doc_id="a", text="doc a", metadata={})
    doc_b = Document(doc_id="b", text="doc b", metadata={})

    retriever = HybridRetriever(
        embedding_retriever=_FixedRetriever([Document(doc_id="a", text="doc a", metadata={})]),
        bm25_retriever=_FixedRetriever([Document(doc_id="b", text="doc b", metadata={})]),
        documents=[doc_a, doc_b],
        top_k=1,
        rrf_k=10,
    )

    result = retriever.retrieve(Query(text="test query"))

    assert len(result.documents) == 1
