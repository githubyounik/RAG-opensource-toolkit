"""Proposition-level text processing for indexing.

This processor is an alternative chunking strategy. It first creates regular
base chunks and then asks an LLM to rewrite each chunk into smaller,
fact-focused proposition documents.
"""

from __future__ import annotations

import json
import re
import time

import httpx

from rag_toolkit.core.types import Document, ParsedFile
from rag_toolkit.indexing.base import TextProcessor
from rag_toolkit.indexing.document_processor import DocumentProcessor
from rag_toolkit.indexing.prompts import PROPOSITION_SYSTEM_PROMPT as _SYSTEM_PROMPT


class PropositionProcessor(TextProcessor):
    """Turn parsed text into proposition-sized documents with OpenRouter.

    The processor works in two stages:
    1. Use the existing `DocumentProcessor` to make regular base chunks.
    2. Convert each base chunk into one or more proposition documents.

    This keeps the behavior aligned with the current indexing framework,
    making proposition chunking a drop-in alternative to the normal
    chunking strategy.
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

    def __init__(
        self,
        api_key: str,
        *,
        base_chunk_size: int = 1000,
        base_chunk_overlap: int = 200,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int | None = 512,
        max_retries: int = 2,
        retry_delay_seconds: float = 2.0,
        site_url: str = "",
        site_name: str = "RAG Toolkit",
    ) -> None:
        super().__init__()
        self.base_processor = DocumentProcessor(
            chunk_size=base_chunk_size,
            chunk_overlap=base_chunk_overlap,
        )
        self._api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self._site_url = site_url
        self._site_name = site_name

    def _build_payload(self, text: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
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
                        "OpenRouter proposition processing request failed "
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
                    "OpenRouter proposition processing request failed after retries "
                    f"(model={self.model}, attempts={self.max_retries + 1}). "
                    f"Original error: {exc}"
                ) from exc

        raise RuntimeError(
            "OpenRouter proposition processing failed without a successful response."
        ) from last_exception

    def _extract_json_text(self, content: str) -> str:
        """Extract the JSON object from a model response string."""

        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("The model response did not contain a JSON object.")
        return content[start : end + 1]

    def _parse_list_propositions(self, content: str) -> list[str]:
        """Fallback parser for numbered or bulleted proposition lists.

        Some chat models ignore the strict JSON instruction and instead return
        explanatory text followed by a numbered list. This parser extracts the
        proposition lines from those responses so the processor remains usable.
        """

        propositions: list[str] = []

        # Match patterns such as:
        # 1. Proposition text
        # - Proposition text
        # * Proposition text
        for line in content.splitlines():
            cleaned_line = line.strip()
            if not cleaned_line:
                continue

            numbered_match = re.match(r"^\d+\.\s+(.*)$", cleaned_line)
            bulleted_match = re.match(r"^[-*]\s+(.*)$", cleaned_line)

            if numbered_match:
                proposition = numbered_match.group(1).strip()
                if proposition:
                    propositions.append(proposition)
                continue

            if bulleted_match:
                proposition = bulleted_match.group(1).strip()
                if proposition:
                    propositions.append(proposition)

        return propositions

    def _parse_propositions(self, content: str) -> list[str]:
        """Parse the JSON response and return cleaned proposition strings."""

        try:
            raw_json = self._extract_json_text(content)
            parsed = json.loads(raw_json)
            propositions = parsed.get("propositions", [])
        except (ValueError, json.JSONDecodeError):
            propositions = self._parse_list_propositions(content)

        if not isinstance(propositions, list):
            raise ValueError("The `propositions` field must be a list.")

        cleaned_propositions: list[str] = []
        for proposition in propositions:
            cleaned = str(proposition).strip()
            if cleaned:
                cleaned_propositions.append(cleaned)

        return cleaned_propositions

    def _process_base_document(self, document: Document) -> list[Document]:
        """Convert one regular chunk into proposition-sized documents."""

        payload = self._build_payload(document.text)
        response_json = self._request_completion(payload)
        content = response_json["choices"][0]["message"]["content"]
        propositions = self._parse_propositions(content)

        proposition_documents: list[Document] = []
        for proposition_index, proposition_text in enumerate(propositions):
            proposition_documents.append(
                Document(
                    doc_id=f"{document.doc_id}-prop-{proposition_index}",
                    text=proposition_text,
                    metadata={
                        **document.metadata,
                        "parent_doc_id": document.doc_id,
                        "proposition_id": proposition_index,
                        "chunking_strategy": "proposition",
                    },
                )
            )

        return proposition_documents

    def process(self, parsed_file: ParsedFile) -> list[Document]:
        """Turn a parsed file into proposition-sized documents."""

        base_documents = self.base_processor.process(parsed_file)

        proposition_documents: list[Document] = []
        for document in base_documents:
            proposition_documents.extend(self._process_base_document(document))

        return proposition_documents
