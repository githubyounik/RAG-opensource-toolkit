"""LLM generation using ZhipuAI GLM-4.7 via the zai-sdk."""

from __future__ import annotations

from rag_toolkit.core.types import GenerationResult, RetrievalResult
from rag_toolkit.generation.base import Generator

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided "
    "context. If the context does not contain enough information to answer, say so "
    "clearly instead of guessing."
)


class ZhipuGenerator(Generator):
    """Generate answers with GLM-4.5 via ZhipuAI.

    Parameters
    ----------
    api_key:
        Your ZhipuAI API key.
    model:
        Model name. Defaults to ``"glm-4.7"``.
    temperature:
        Sampling temperature (0–1). Defaults to ``0.6``.
    max_tokens:
        Maximum tokens in the generated answer. ``None`` uses the API default.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4.7",
        temperature: float = 0.6,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        try:
            from zai import ZhipuAiClient
        except ImportError as exc:
            raise ImportError(
                "zai-sdk is required for ZhipuGenerator. "
                "Install it with: pip install zai-sdk"
            ) from exc

        self._client = ZhipuAiClient(api_key=api_key)
        self.model = model
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
        """Call GLM-4.5 and return a :class:`GenerationResult`."""
        prompt = self._build_prompt(result)

        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self._client.chat.completions.create(**kwargs)
        answer = response.choices[0].message.content

        return GenerationResult(
            query=result.query,
            answer=answer,
            contexts=list(result.documents),
            metadata={
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            },
        )
