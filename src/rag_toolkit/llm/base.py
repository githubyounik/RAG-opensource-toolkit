"""Shared abstractions for chat-based LLM providers.

This module keeps the provider-facing API intentionally small. The business
layers in this project, such as generation and pre-retrieval, only need two
things from an LLM call:

1. The returned text
2. Basic token usage metadata when the provider exposes it

By standardizing that contract here, the higher-level components can reuse the
same provider clients without caring whether the request goes to OpenRouter or
Zhipu.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """Normalized text response returned by a chat provider."""

    text: str
    usage: dict[str, int | None] = field(default_factory=dict)


class ChatLLMClient(ABC):
    """Base interface for chat-based LLM providers."""

    @abstractmethod
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
    ) -> LLMResponse:
        """Run one chat completion request and return normalized text."""
