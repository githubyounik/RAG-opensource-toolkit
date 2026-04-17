import pytest

from rag_toolkit.core.types import Document
from rag_toolkit.embeddings.vector_index import VectorIndex
from rag_toolkit.retrieval import (
    BM25Retriever,
    EmbeddingRetriever,
    HybridRetriever,
    create_retriever_from_config,
)


class _StubEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return [float(len(text))]


def test_retrieval_factory_creates_embedding_retriever() -> None:
    documents = [Document(doc_id="doc-1", text="alpha", metadata={})]
    index = VectorIndex()
    index.add(documents[0], [1.0])

    retriever = create_retriever_from_config(
        {"strategy": "embedding", "embedding": {"top_k": 2}},
        documents=documents,
        index=index,
        embedder=_StubEmbedder(),
    )

    assert isinstance(retriever, EmbeddingRetriever)
    assert retriever.top_k == 2


def test_retrieval_factory_creates_bm25_retriever() -> None:
    documents = [Document(doc_id="doc-1", text="alpha beta", metadata={})]

    retriever = create_retriever_from_config(
        {"strategy": "bm25", "bm25": {"top_k": 3, "k1": 1.3, "b": 0.6}},
        documents=documents,
    )

    assert isinstance(retriever, BM25Retriever)
    assert retriever.top_k == 3
    assert retriever.k1 == 1.3
    assert retriever.b == 0.6


def test_retrieval_factory_creates_hybrid_retriever() -> None:
    documents = [Document(doc_id="doc-1", text="alpha beta", metadata={})]
    index = VectorIndex()
    index.add(documents[0], [1.0])

    retriever = create_retriever_from_config(
        {
            "strategy": "hybrid",
            "embedding": {"top_k": 5},
            "bm25": {"top_k": 6},
            "hybrid": {"top_k": 3, "rrf_k": 42},
        },
        documents=documents,
        index=index,
        embedder=_StubEmbedder(),
    )

    assert isinstance(retriever, HybridRetriever)
    assert retriever.top_k == 3
    assert retriever.rrf_k == 42


def test_retrieval_factory_requires_index_for_embedding() -> None:
    documents = [Document(doc_id="doc-1", text="alpha", metadata={})]

    with pytest.raises(ValueError, match="VectorIndex"):
        create_retriever_from_config(
            {"strategy": "embedding"},
            documents=documents,
            embedder=_StubEmbedder(),
        )


def test_retrieval_factory_requires_index_for_hybrid() -> None:
    documents = [Document(doc_id="doc-1", text="alpha", metadata={})]

    with pytest.raises(ValueError, match="VectorIndex"):
        create_retriever_from_config(
            {"strategy": "hybrid"},
            documents=documents,
            embedder=_StubEmbedder(),
        )
