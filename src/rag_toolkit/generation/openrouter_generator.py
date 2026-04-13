"""LLM generation using a shared OpenRouter chat client."""

from __future__ import annotations

from rag_toolkit.core.types import GenerationResult, RetrievalResult
from rag_toolkit.generation.base import Generator
from rag_toolkit.llm import OpenRouterChatClient

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided "
    "context. If the context does not contain enough information to answer, say so "
    "clearly instead of guessing."
)


class OpenRouterGenerator(Generator):
    """Generate answers with an OpenRouter chat model."""

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
        self._client = OpenRouterChatClient(
            api_key=api_key,
            model=model,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            site_url=site_url,
            site_name=site_name,
        )
        self.model = self._client.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

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

    def generate(self, result: RetrievalResult) -> GenerationResult:
        """Call OpenRouter chat completions and return a GenerationResult."""
        prompt = self._build_prompt(result)
        response = self._client.complete(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return GenerationResult(
            query=result.query,
            answer=response.text,
            contexts=list(result.documents),
            metadata={
                "provider": "openrouter",
                "model": self.model,
                "usage": dict(response.usage),
            },
        )
