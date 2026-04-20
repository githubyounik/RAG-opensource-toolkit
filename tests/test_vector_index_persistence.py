import importlib.util

import pytest

from rag_toolkit.core.types import Document
from rag_toolkit.embeddings import EmbeddingIndexer, VectorIndex


FAISS_AVAILABLE = importlib.util.find_spec("faiss") is not None


class _StubEmbedder:
    model = "stub-embedder"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]

    def embed_one(self, text: str) -> list[float]:
        return [1.0, float(len(text))]


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss is not installed in this environment")
def test_vector_index_save_and_load_roundtrip(tmp_path) -> None:
    index = VectorIndex()
    index.add(Document(doc_id="doc-1", text="alpha", metadata={"source": "test"}), [1.0, 0.0])
    index.add(Document(doc_id="doc-2", text="beta", metadata={"source": "test"}), [0.0, 1.0])
    index.build_faiss()

    target_dir = tmp_path / "index"
    index.save(str(target_dir))

    loaded = VectorIndex.load(str(target_dir))

    assert len(loaded) == 2
    assert loaded.documents[0].doc_id == "doc-1"
    assert loaded.faiss_index is not None


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss is not installed in this environment")
def test_embedding_indexer_build_or_load_reuses_existing_index(tmp_path) -> None:
    documents = [
        Document(doc_id="doc-1", text="alpha", metadata={}),
        Document(doc_id="doc-2", text="beta", metadata={}),
    ]
    indexer = EmbeddingIndexer(_StubEmbedder())

    first_index, index_dir, first_loaded = indexer.build_or_load(
        documents,
        cache_dir=str(tmp_path),
        reuse_existing=True,
        namespace="demo",
    )
    second_index, second_dir, second_loaded = indexer.build_or_load(
        documents,
        cache_dir=str(tmp_path),
        reuse_existing=True,
        namespace="demo",
    )

    assert len(first_index) == 2
    assert index_dir == second_dir
    assert first_loaded is False
    assert second_loaded is True
    assert second_index.documents[1].doc_id == "doc-2"
