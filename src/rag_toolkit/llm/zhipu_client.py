"""Shared Zhipu client used by multiple business modules."""

from __future__ import annotations

from rag_toolkit.llm.base import ChatLLMClient, LLMResponse


class ZhipuChatClient(ChatLLMClient):
    """Call Zhipu chat completions and normalize the output."""

    def __init__(self, api_key: str, *, model: str) -> None:
        try:
            from zai import ZhipuAiClient
        except ImportError as exc:
            raise ImportError(
                "zai-sdk is required for Zhipu support. "
                "Install it with: pip install zai-sdk"
            ) from exc

        self._client = ZhipuAiClient(api_key=api_key)
        self.model = model

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResponse:
        """Run one Zhipu chat completion and normalize the output."""

        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**kwargs)

        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        )
