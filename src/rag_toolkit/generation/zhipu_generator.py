"""LLM generation using a shared Zhipu chat client."""

from __future__ import annotations

from rag_toolkit.core.types import GenerationResult, RetrievalResult
from rag_toolkit.generation.base import Generator
from rag_toolkit.llm import ZhipuChatClient

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided "
    "context. If the context does not contain enough information to answer, say so "
    "clearly instead of guessing."
)


class ZhipuGenerator(Generator):
    """Generate answers with GLM models via ZhipuAI."""

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4.7",
        temperature: float = 0.6,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self._client = ZhipuChatClient(api_key=api_key, model=model)
        self.model = self._client.model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_prompt(self, result: RetrievalResult) -> str:
        if not result.documents:
            context_text = "(no context retrieved)"
        else:
            blocks = [
                f"[{i + 1}] {doc.text}" for i, doc in enumerate(result.documents)
            ]
            context_text = "\n\n".join(blocks)

        return (
            f"Context:\n{context_text}\n\n"
            f"Question: {result.query.text}\n\n"
            "Answer:"
        )

    def generate(self, result: RetrievalResult) -> GenerationResult:
        """Call Zhipu chat completions and return a :class:`GenerationResult`."""
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
                "provider": "zhipu",
                "model": self.model,
                "usage": dict(response.usage),
            },
        )
