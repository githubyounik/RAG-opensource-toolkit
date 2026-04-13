"""Reusable LLM-backed transformer for pre-retrieval query changes."""

from __future__ import annotations

from rag_toolkit.llm import ChatLLMClient


class QueryTransformer:
    """Call a shared LLM client and return transformed query text.

    The pre-retrieval layer only needs a small wrapper around the shared client:
    provide a system prompt, provide the original user query, and get back plain
    text. Keeping this helper separate makes the pre-retrieval classes easier to
    read while still reusing the lower-level provider logic.
    """

    def __init__(
        self,
        client: ChatLLMClient,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = 256,
    ) -> None:
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def transform(self, *, system_prompt: str, user_prompt: str) -> str:
        """Transform query text with the configured provider."""

        response = self.client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.text.strip()
