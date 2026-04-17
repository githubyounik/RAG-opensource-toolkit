from rag_toolkit.core.types import Document, Query
from rag_toolkit.retrieval import BM25Retriever


def test_bm25_retriever_ranks_keyword_match_first() -> None:
    documents = [
        Document(doc_id="doc-1", text="apple banana", metadata={"source": "test"}),
        Document(doc_id="doc-2", text="banana banana orange", metadata={"source": "test"}),
        Document(doc_id="doc-3", text="grape pear", metadata={"source": "test"}),
    ]

    retriever = BM25Retriever(documents=documents, top_k=2)
    result = retriever.retrieve(Query(text="banana orange"))

    assert [document.doc_id for document in result.documents] == ["doc-2", "doc-1"]
    assert result.metadata["retrieval_strategy"] == "bm25"
    assert result.documents[0].metadata["score"] >= result.documents[1].metadata["score"]


def test_bm25_retriever_returns_zero_scores_for_empty_query() -> None:
    documents = [
        Document(doc_id="doc-1", text="alpha beta", metadata={}),
        Document(doc_id="doc-2", text="gamma delta", metadata={}),
    ]

    retriever = BM25Retriever(documents=documents, top_k=2)
    result = retriever.retrieve(Query(text=""))

    assert len(result.documents) == 2
    assert all(document.metadata["score"] == 0.0 for document in result.documents)
