"""LLM generation using OpenRouter chat completions."""

from __future__ import annotations

import time

import httpx

from rag_toolkit.core.types import GenerationResult, RetrievalResult
from rag_toolkit.generation.base import Generator

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided "
    "context. If the context does not contain enough information to answer, say so "
    "clearly instead of guessing."
)


class OpenRouterGenerator(Generator):
    """Generate answers with an OpenRouter chat model.

    Parameters
    ----------
    api_key:
        Your OpenRouter API key.
    model:
        Full OpenRouter model id.
    temperature:
        Sampling temperature (0-1). Defaults to ``0.6``.
    max_tokens:
        Maximum tokens in the generated answer. ``None`` uses the API default.
    max_retries:
        Number of retries after the first failed request. Defaults to ``2``.
    retry_delay_seconds:
        Delay between retries in seconds. Defaults to ``2.0``.
    site_url:
        Optional site URL sent to OpenRouter for attribution.
    site_name:
        Optional site name sent to OpenRouter for attribution.
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.6,
        max_tokens: int | None = None,
        max_retries: int = 2,
        retry_delay_seconds: float = 2.0,
        site_url: str = "",
        site_name: str = "RAG Toolkit",
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._site_url = site_url
        self._site_name = site_name

    def _build_prompt(self, result: RetrievalResult) -> str:
        if not result.documents:
            context_text = "(no context retrieved)"
        else:
            blocks = [f"[{i + 1}] {doc.text}" for i, doc in enumerate(result.documents)]
            context_text = "\n\n".join(blocks)

        return (
            f"Context:\n{context_text}\n\n"
            f"Question: {result.query.text}\n\n"
            "Answer:"
        )

    def _build_payload(self, prompt: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def _should_retry_status(self, status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code < 600

    def _request_completion(self, payload: dict[str, object]) -> dict[str, object]:
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": self._site_url,
                        "X-OpenRouter-Title": self._site_name,
                    },
                    json=payload,
                    timeout=60,
                )

                if response.is_error:
                    status_code = response.status_code
                    response_preview = response.text[:500]

                    if attempt < self.max_retries and self._should_retry_status(status_code):
                        time.sleep(self.retry_delay_seconds)
                        continue

                    raise RuntimeError(
                        "OpenRouter generation request failed "
                        f"(status={status_code}, model={self.model}, attempt={attempt + 1}/"
                        f"{self.max_retries + 1}). Response: {response_preview}"
                    )

                return response.json()

            except httpx.RequestError as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)
                    continue

                raise RuntimeError(
                    "OpenRouter generation request failed after retries "
                    f"(model={self.model}, attempts={self.max_retries + 1}). "
                    f"Original error: {exc}"
                ) from exc

        raise RuntimeError(
            "OpenRouter generation failed without a successful response."
        ) from last_exception

    def generate(self, result: RetrievalResult) -> GenerationResult:
        """Call OpenRouter chat completions and return a GenerationResult."""
        prompt = self._build_prompt(result)
        payload = self._build_payload(prompt)
        response_json = self._request_completion(payload)

        answer = response_json["choices"][0]["message"]["content"]
        usage = response_json.get("usage", {})

        return GenerationResult(
            query=result.query,
            answer=answer,
            contexts=list(result.documents),
            metadata={
                "provider": "openrouter",
                "model": self.model,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                },
            },
        )
