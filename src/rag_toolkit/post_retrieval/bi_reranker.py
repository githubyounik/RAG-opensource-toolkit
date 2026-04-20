"""Local bi-encoder reranking using a HuggingFace encoder model."""

from __future__ import annotations

import numpy as np

from rag_toolkit.core.types import Document, RetrievalResult
from rag_toolkit.post_retrieval.base import PostRetriever


class BiReranker(PostRetriever):
    """Rerank retrieved chunks using a local bi-encoder model.

    A bi-encoder encodes the query and each document separately, then
    ranks by dot-product similarity between the normalised embeddings.
    This is faster than a cross-encoder but typically less precise.

    Parameters
    ----------
    model_path:
        HuggingFace model ID or local path to a transformer encoder,
        e.g. ``BAAI/bge-reranker-base``.
    top_k:
        Maximum number of documents to keep after reranking.  ``None``
        keeps all documents.
    max_length:
        Maximum token length for each encoded text.
    batch_size:
        Number of texts sent to the model per forward pass.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU.
        Pass ``"cpu"`` or ``"cuda"`` to override.
    pooling_method:
        How to aggregate token embeddings into a single vector.
        ``"mean"`` averages non-padding tokens; ``"cls"`` takes the
        first token.
    """

    def __init__(
        self,
        model_path: str,
        *,
        top_k: int | None = None,
        max_length: int = 512,
        batch_size: int = 32,
        device: str = "auto",
        pooling_method: str = "mean",
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for BiReranker. "
                "Install them with: pip install torch transformers"
            ) from exc

        super().__init__()
        self.model_path = model_path
        self.top_k = top_k
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling_method = pooling_method

        self._device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path)
        self._model.eval()
        self._model.to(self._device)

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode *texts* into L2-normalised embeddings of shape ``(N, D)``."""
        import torch

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self._device)
            with torch.inference_mode():
                output = self._model(**inputs, return_dict=True)

            if self.pooling_method == "cls":
                emb = output.last_hidden_state[:, 0, :]
            else:
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                emb = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embeddings.append(emb.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

    def process(self, result: RetrievalResult) -> RetrievalResult:
        """Rerank retrieved documents using bi-encoder dot-product scores."""

        if not result.documents:
            return result

        query_emb = self._encode([result.query.text])  # (1, D)
        doc_emb = self._encode([doc.text for doc in result.documents])  # (N, D)
        scores: list[float] = (query_emb @ doc_emb.T).flatten().tolist()

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
                        "rerank_score": float(score),
                        "post_retrieval_strategy": "bi_rerank",
                        "rerank_model": self.model_path,
                    },
                )
            )

        return RetrievalResult(
            query=result.query,
            documents=reranked_documents,
            metadata={
                **result.metadata,
                "post_retrieval_strategy": "bi_rerank",
                "rerank_model": self.model_path,
                "original_retrieved_count": len(result.documents),
                "reranked_count": len(reranked_documents),
            },
        )
