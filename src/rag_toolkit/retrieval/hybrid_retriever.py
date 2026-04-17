"""Hybrid retriever that fuses dense and sparse rankings with RRF."""

from __future__ import annotations

from rag_toolkit.core.types import Document, Query, RetrievalResult
from rag_toolkit.retrieval.base import Retriever


class HybridRetriever(Retriever):
    """Fuse embedding and BM25 retrieval results with Reciprocal Rank Fusion.

    Parameters
    ----------
    embedding_retriever:
        Dense retriever used to produce one ranked list.
    bm25_retriever:
        Sparse retriever used to produce the second ranked list.
    documents:
        Full indexed document list, used as the canonical text/metadata store
        when the same ``doc_id`` appears in both rankings.
    top_k:
        Number of fused documents to return.
    rrf_k:
        Reciprocal Rank Fusion constant. Larger values flatten rank influence.
    """

    def __init__(
        self,
        *,
        embedding_retriever: Retriever,
        bm25_retriever: Retriever,
        documents: list[Document],
        top_k: int = 4,
        rrf_k: int = 60,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")

        self.embedding_retriever = embedding_retriever
        self.bm25_retriever = bm25_retriever
        self.documents = documents
        self.top_k = top_k
        self.rrf_k = rrf_k
        self._documents_by_id = {document.doc_id: document for document in documents}

    def _fuse_rankings(
        self,
        *,
        embedding_result: RetrievalResult,
        bm25_result: RetrievalResult,
    ) -> list[Document]:
        fused_scores: dict[str, float] = {}
        fused_metadata: dict[str, dict[str, object]] = {}

        for source_name, documents in (
            ("embedding", embedding_result.documents),
            ("bm25", bm25_result.documents),
        ):
            for rank, document in enumerate(documents, start=1):
                doc_id = document.doc_id
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )

                metadata = fused_metadata.setdefault(
                    doc_id,
                    {
                        "embedding_rank": None,
                        "embedding_score": None,
                        "bm25_rank": None,
                        "bm25_score": None,
                    },
                )
                metadata[f"{source_name}_rank"] = rank
                metadata[f"{source_name}_score"] = document.metadata.get("score")

        ranked_doc_ids = sorted(
            fused_scores,
            key=lambda doc_id: fused_scores[doc_id],
            reverse=True,
        )

        fused_documents: list[Document] = []
        for doc_id in ranked_doc_ids[: self.top_k]:
            base_document = self._documents_by_id.get(doc_id)
            if base_document is None:
                continue

            fused_documents.append(
                Document(
                    doc_id=base_document.doc_id,
                    text=base_document.text,
                    metadata={
                        **base_document.metadata,
                        **fused_metadata.get(doc_id, {}),
                        "score": fused_scores[doc_id],
                        "retrieval_strategy": "hybrid_rrf",
                    },
                )
            )

        return fused_documents

    def retrieve(self, query: Query) -> RetrievalResult:
        """Run dense and sparse retrieval, then fuse the rankings with RRF."""

        embedding_result = self.embedding_retriever.retrieve(query)
        bm25_result = self.bm25_retriever.retrieve(query)
        fused_documents = self._fuse_rankings(
            embedding_result=embedding_result,
            bm25_result=bm25_result,
        )

        return RetrievalResult(
            query=query,
            documents=fused_documents,
            metadata={
                "top_k": self.top_k,
                "rrf_k": self.rrf_k,
                "retrieval_strategy": "hybrid_rrf",
                "embedding_top_k": len(embedding_result.documents),
                "bm25_top_k": len(bm25_result.documents),
            },
        )
