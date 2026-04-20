"""Local cross-encoder reranking using a HuggingFace sequence-classification model."""

from __future__ import annotations

from rag_toolkit.core.types import Document, RetrievalResult
from rag_toolkit.post_retrieval.base import PostRetriever


class CrossReranker(PostRetriever):
    """Rerank retrieved chunks using a local cross-encoder model.

    A cross-encoder tokenises each (query, document) pair together and
    produces a single relevance logit.  This gives higher quality scores
    than a bi-encoder but is slower because each pair must be evaluated
    independently.

    Parameters
    ----------
    model_path:
        HuggingFace model ID or local path to a sequence-classification
        model, e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
    top_k:
        Maximum number of documents to keep after reranking.  ``None``
        keeps all documents.
    max_length:
        Maximum total token length for a (query, document) pair.
    batch_size:
        Number of pairs sent to the model per forward pass.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU.
        Pass ``"cpu"`` or ``"cuda"`` to override.
    """

    def __init__(
        self,
        model_path: str,
        *,
        top_k: int | None = None,
        max_length: int = 512,
        batch_size: int = 32,
        device: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for CrossReranker. "
                "Install them with: pip install torch transformers"
            ) from exc

        super().__init__()
        self.model_path = model_path
        self.top_k = top_k
        self.max_length = max_length
        self.batch_size = batch_size

        self._device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=1
        )
        self._model.eval()
        self._model.to(self._device)

    def _score_pairs(self, query_text: str, documents: list[Document]) -> list[float]:
        """Score every (query, document) pair and return a flat list of floats."""
        import torch

        pairs = [[query_text, doc.text] for doc in documents]
        all_scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self._device)
            with torch.inference_mode():
                logits = (
                    self._model(**inputs, return_dict=True)
                    .logits.view(-1)
                    .float()
                    .cpu()
                )
            all_scores.extend(logits.tolist())
        return all_scores

    def process(self, result: RetrievalResult) -> RetrievalResult:
        """Rerank retrieved documents using cross-encoder relevance scores."""

        if not result.documents:
            return result

        scores = self._score_pairs(result.query.text, result.documents)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        if self.top_k is not None:
            ranked = ranked[: self.top_k]

        reranked_documents: list[Document] = []
        for reranked_rank, (original_index, score) in enumerate(ranked):
            original_document = result.documents[original_index]
            reranked_documents.append(
                Document(
                    doc_id=original_document.doc_id,
                    text=original_document.text,
                    metadata={
                        **original_document.metadata,
                        "original_rank": original_index,
                        "reranked_rank": reranked_rank,
                        "rerank_score": score,
                        "post_retrieval_strategy": "cross_rerank",
                        "rerank_model": self.model_path,
                    },
                )
            )

        return RetrievalResult(
            query=result.query,
            documents=reranked_documents,
            metadata={
                **result.metadata,
                "post_retrieval_strategy": "cross_rerank",
                "rerank_model": self.model_path,
                "original_retrieved_count": len(result.documents),
                "reranked_count": len(reranked_documents),
            },
        )
