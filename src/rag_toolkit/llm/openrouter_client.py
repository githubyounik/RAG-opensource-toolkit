"""Shared OpenRouter client used by multiple business modules."""

from __future__ import annotations

import time

import httpx

from rag_toolkit.llm.base import ChatLLMClient, LLMResponse


class OpenRouterChatClient(ChatLLMClient):
    """Call OpenRouter chat completions with shared retry logic."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        max_retries: int = 2,
        retry_delay_seconds: float = 2.0,
        site_url: str = "",
        site_name: str = "RAG Toolkit",
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._site_url = site_url
        self._site_name = site_name

    def _should_retry_status(self, status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code < 600

    def _build_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def _extract_message_text(self, response_json: dict[str, object]) -> str:
        """Extract plain text content from an OpenRouter chat response.

        Some models can return ``content = null`` when the completion is cut off,
        when the model emitted reasoning but no final answer text, or when the
        provider uses a slightly different response shape. In that case we raise
        a clear error instead of failing later with an unhelpful attribute error.
        """

        try:
            message = response_json["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "OpenRouter response did not contain a standard message.content field "
                f"(model={self.model})."
            ) from exc

        if isinstance(content, str):
            return content.strip()

        finish_reason = response_json.get("choices", [{}])[0].get("finish_reason")
        usage = response_json.get("usage", {})
        reasoning_tokens = None
        if isinstance(usage, dict):
            completion_details = usage.get("completion_tokens_details", {})
            if isinstance(completion_details, dict):
                reasoning_tokens = completion_details.get("reasoning_tokens")

        raise RuntimeError(
            "OpenRouter response contained no text content "
            f"(model={self.model}, finish_reason={finish_reason}, content_type="
            f"{type(content).__name__}, reasoning_tokens={reasoning_tokens}). "
            "The model may have spent tokens on reasoning without producing a final "
            "text answer. Try increasing max_tokens or switching to a model that "
            "returns plain text content."
        )

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResponse:
        """Run one OpenRouter chat completion and normalize the output."""

        payload = self._build_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
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
                        "OpenRouter request failed "
                        f"(status={status_code}, model={self.model}, attempt={attempt + 1}/"
                        f"{self.max_retries + 1}). Response: {response_preview}"
                    )

                response_json = response.json()
                message = self._extract_message_text(response_json)
                usage = response_json.get("usage", {})

                return LLMResponse(
                    text=message,
                    usage={
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                    },
                )

            except httpx.RequestError as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)
                    continue

                raise RuntimeError(
                    "OpenRouter request failed after retries "
                    f"(model={self.model}, attempts={self.max_retries + 1}). "
                    f"Original error: {exc}"
                ) from exc

        raise RuntimeError(
            "OpenRouter request failed without a successful response."
        ) from last_exception
