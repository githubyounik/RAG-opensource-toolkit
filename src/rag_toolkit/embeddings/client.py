"""OpenRouter embedding client."""

from __future__ import annotations

import httpx

from rag_toolkit.embeddings.base import TextEmbedder


class OpenRouterEmbedder(TextEmbedder):
    """Calls embedding models via OpenRouter's REST API.

    Uses multimodal content format required by models like
    nvidia/llama-nemotron-embed-vl-1b-v2:free.
    """

    DEFAULT_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    API_URL = "https://openrouter.ai/api/v1/embeddings"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        site_url: str = "",
        site_name: str = "RAG Toolkit",
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self.model = model
        self._site_url = site_url
        self._site_name = site_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""
        # Multimodal content format (text-only) required by this model family
        inputs = [{"content": [{"type": "text", "text": t}]} for t in texts]

        response = httpx.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self._site_url,
                "X-OpenRouter-Title": self._site_name,
            },
            json={"model": self.model, "input": inputs, "encoding_format": "float"},
            timeout=60,
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for a single text."""
        return self.embed([text])[0]
