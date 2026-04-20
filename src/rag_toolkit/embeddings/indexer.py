"""Embedding indexer: converts Documents into a searchable VectorIndex."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag_toolkit.core.types import Document
from rag_toolkit.embeddings.base import TextEmbedder, VectorIndexBuilder
from rag_toolkit.embeddings.vector_index import VectorIndex


class EmbeddingIndexer(VectorIndexBuilder):
    """Embed a list of Documents and store them in a VectorIndex.

    Parameters
    ----------
    embedder:
        Any :class:`TextEmbedder` implementation.
    batch_size:
        Number of documents sent to the embedding API per request.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.batch_size = batch_size

    def build(self, documents: list[Document]) -> VectorIndex:
        """Embed *documents* and return a populated VectorIndex."""
        index = VectorIndex()
        for batch_start in range(0, len(documents), self.batch_size):
            batch = documents[batch_start : batch_start + self.batch_size]
            embeddings = self.embedder.embed([doc.text for doc in batch])
            for doc, emb in zip(batch, embeddings):
                index.add(doc, emb)
        index.build_faiss()
        return index

    def build_or_load(
        self,
        documents: list[Document],
        *,
        cache_dir: str,
        reuse_existing: bool = True,
        namespace: str | None = None,
    ) -> tuple[VectorIndex, str, bool]:
        """Load a persisted FAISS index when possible, otherwise build and save one.

        Returns ``(index, directory, loaded_from_disk)``.
        """

        index_directory = self._index_directory(
            documents,
            cache_dir=cache_dir,
            namespace=namespace,
        )
        if reuse_existing and VectorIndex.exists(index_directory):
            return VectorIndex.load(index_directory), index_directory, True

        index = self.build(documents)
        index.save(index_directory)
        return index, index_directory, False

    def lookup_cached_index_for_file(
        self,
        file_path: str,
        *,
        cache_dir: str,
    ) -> str | None:
        """Return a cached index directory for *file_path* if a matching fingerprint exists."""
        registry = self._load_registry(cache_dir)
        file_fingerprint = self.file_fingerprint(file_path)
        index_directory = registry.get(file_fingerprint)
        if index_directory and VectorIndex.exists(index_directory):
            return index_directory
        return None

    def register_cached_index_for_file(
        self,
        file_path: str,
        *,
        cache_dir: str,
        index_directory: str,
    ) -> None:
        """Record that *file_path* has been indexed at *index_directory*."""
        registry = self._load_registry(cache_dir)
        registry[self.file_fingerprint(file_path)] = index_directory
        registry_path = self._registry_path(cache_dir)
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        with registry_path.open("w", encoding="utf-8") as file:
            json.dump(registry, file, ensure_ascii=False, indent=2)

    def _index_directory(
        self,
        documents: list[Document],
        *,
        cache_dir: str,
        namespace: str | None = None,
    ) -> str:
        cache_root = Path(cache_dir)
        safe_namespace = namespace or "default"
        fingerprint = self._fingerprint_documents(documents)
        return str(cache_root / safe_namespace / fingerprint)

    def _fingerprint_documents(self, documents: list[Document]) -> str:
        payload = {
            "embedder_model": getattr(self.embedder, "model", self.embedder.__class__.__name__),
            "documents": [
                {
                    "doc_id": document.doc_id,
                    "text": document.text,
                    "metadata": document.metadata,
                }
                for document in documents
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def file_fingerprint(self, file_path: str) -> str:
        """Return a content fingerprint for the original source file."""
        path = Path(file_path)
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    def _registry_path(self, cache_dir: str) -> Path:
        return Path(cache_dir) / "file_registry.json"

    def _load_registry(self, cache_dir: str) -> dict[str, str]:
        registry_path = self._registry_path(cache_dir)
        if not registry_path.exists():
            return {}
        with registry_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            return {}
        return {str(key): str(value) for key, value in data.items()}
