"""Vector index that stores document chunks alongside their embeddings."""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rag_toolkit.core.types import Document


@dataclass(slots=True)
class VectorIndex:
    """In-memory store of documents and their embedding vectors.

    ``documents`` and ``embeddings`` are kept in sync: ``embeddings[i]`` is
    the embedding for ``documents[i]``.
    """

    documents: list[Document] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    faiss_index: object | None = None
    normalize: bool = True

    def add(self, document: Document, embedding: list[float]) -> None:
        """Append a document and its embedding together."""
        self.documents.append(document)
        self.embeddings.append(embedding)

    def build_faiss(self) -> None:
        """Build an in-memory FAISS index from the stored embeddings."""
        if not self.embeddings:
            self.faiss_index = None
            return

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for FAISS-backed vector search and persistence."
            ) from exc

        embedding_matrix = np.asarray(self.embeddings, dtype=np.float32)
        if embedding_matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix to build a FAISS index.")

        if self.normalize:
            embedding_matrix = embedding_matrix.copy()
            faiss.normalize_L2(embedding_matrix)

        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        index.add(embedding_matrix)
        self.faiss_index = index

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[float, Document]]:
        """Search the FAISS index and return scored documents."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.faiss_index is None:
            self.build_faiss()
        if self.faiss_index is None:
            return []

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for FAISS-backed vector search and persistence."
            ) from exc

        query = np.asarray([query_embedding], dtype=np.float32)
        if self.normalize:
            query = query.copy()
            faiss.normalize_L2(query)

        scores, indexes = self.faiss_index.search(query, min(top_k, len(self.documents)))
        results: list[tuple[float, Document]] = []
        for score, index in zip(scores[0], indexes[0]):
            if index < 0:
                continue
            document = self.documents[index]
            scored_document = Document(
                doc_id=document.doc_id,
                text=document.text,
                metadata={**document.metadata, "score": float(score)},
            )
            results.append((float(score), scored_document))
        return results

    def save(self, directory: str) -> None:
        """Persist the FAISS index and aligned documents to disk."""
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for FAISS-backed vector search and persistence."
            ) from exc

        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.faiss_index is None:
            self.build_faiss()

        if self.faiss_index is None:
            raise ValueError("Cannot save an empty FAISS index.")

        faiss.write_index(self.faiss_index, str(target_dir / "index.faiss"))
        if self.embeddings:
            np.save(
                target_dir / "embeddings.npy",
                np.asarray(self.embeddings, dtype=np.float32),
            )
        with (target_dir / "documents.json").open("w", encoding="utf-8") as file:
            json.dump([asdict(document) for document in self.documents], file, ensure_ascii=False)
        with (target_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "document_count": len(self.documents),
                    "embedding_count": len(self.embeddings),
                    "normalize": self.normalize,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str) -> VectorIndex:
        """Load a persisted FAISS index and aligned documents from disk."""
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for FAISS-backed vector search and persistence."
            ) from exc

        target_dir = Path(directory)
        with (target_dir / "documents.json").open("r", encoding="utf-8") as file:
            raw_documents = json.load(file)
        with (target_dir / "metadata.json").open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        documents = [Document(**raw_document) for raw_document in raw_documents]
        embeddings_path = target_dir / "embeddings.npy"
        embeddings: list[list[float]] = []
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path).tolist()
        index = cls(
            documents=documents,
            embeddings=embeddings,
            faiss_index=faiss.read_index(str(target_dir / "index.faiss")),
            normalize=bool(metadata.get("normalize", True)),
        )
        return index

    @staticmethod
    def exists(directory: str) -> bool:
        """Return whether a persisted FAISS index exists at *directory*."""
        target_dir = Path(directory)
        return (
            (target_dir / "index.faiss").exists()
            and (target_dir / "documents.json").exists()
            and (target_dir / "metadata.json").exists()
        )

    @classmethod
    def load_all(cls, root_directory: str) -> VectorIndex:
        """Load and merge every persisted index found under *root_directory*."""
        root = Path(root_directory)
        if not root.exists():
            raise FileNotFoundError(f"Index cache directory does not exist: {root_directory}")

        merged = cls()
        index_directories = sorted(
            {
                path.parent
                for path in root.rglob("index.faiss")
                if cls.exists(str(path.parent))
            }
        )
        if not index_directories:
            raise FileNotFoundError(f"No persisted FAISS indexes found under: {root_directory}")

        for index_directory in index_directories:
            loaded = cls.load(str(index_directory))
            if not loaded.embeddings:
                continue
            merged.documents.extend(loaded.documents)
            merged.embeddings.extend(loaded.embeddings)

        if not merged.embeddings:
            raise ValueError(
                "Persisted indexes were found, but none contained saved embeddings. "
                "Rebuild the indexes with the current persistence format first."
            )

        merged.build_faiss()
        return merged

    def __len__(self) -> int:
        return len(self.documents)
