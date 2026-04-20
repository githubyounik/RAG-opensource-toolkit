"""Local embedding client backed by a HuggingFace encoder model."""

from __future__ import annotations

import numpy as np

from rag_toolkit.embeddings.base import TextEmbedder


class LocalEmbedder(TextEmbedder):
    """Embed texts using a local HuggingFace encoder model.

    The model is loaded once at construction time and kept in memory.
    Any standard AutoModel-compatible encoder can be used, e.g.
    ``BAAI/bge-small-en-v1.5`` or ``sentence-transformers/all-MiniLM-L6-v2``.

    Parameters
    ----------
    model:
        HuggingFace model ID or absolute local path.
    max_length:
        Maximum token length per input text.
    batch_size:
        Number of texts encoded per forward pass.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU.
        Pass ``"cpu"`` or ``"cuda"`` to override.
    pooling_method:
        How to aggregate per-token embeddings into one vector.
        ``"mean"`` averages non-padding tokens; ``"cls"`` takes the
        first ``[CLS]`` token.
    """

    def __init__(
        self,
        model: str,
        *,
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
                "torch and transformers are required for LocalEmbedder. "
                "Install them with: pip install torch transformers"
            ) from exc

        super().__init__()
        self.model = model
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling_method = pooling_method

        self._device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model)
        self._model.eval()
        self._model.to(self._device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalised embedding vector per input text."""
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

        return np.concatenate(all_embeddings, axis=0).tolist()
