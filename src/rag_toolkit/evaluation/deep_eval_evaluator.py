"""DeepEval-style evaluation metrics implemented inside the toolkit.

This module mirrors the core ideas shown in the reference notebook:

1. Correctness: compare the generated answer with a reference answer
2. Faithfulness: check whether the answer is grounded in retrieved context
3. Contextual relevancy: check whether retrieved context is relevant to query

Instead of depending directly on the ``deepeval`` package, this component
reuses the toolkit's shared LLM client abstraction so it plugs cleanly into the
existing modular architecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_toolkit.core.types import EvaluationResult, GenerationResult
from rag_toolkit.evaluation.base import Evaluator
from rag_toolkit.llm import ChatLLMClient

_CORRECTNESS_PROMPT = """
You are evaluating answer correctness.

Compare the generated answer against the reference answer. Focus on factual
agreement, not wording style. Return a score between 0 and 1.

Return exactly in this format:
SCORE: <number between 0 and 1>
REASON: <one short sentence>
""".strip()

_FAITHFULNESS_PROMPT = """
You are evaluating answer faithfulness for a RAG system.

Decide how well the generated answer is grounded in the retrieved context.
If the answer contains claims not supported by the context, reduce the score.
Return a score between 0 and 1.

Return exactly in this format:
SCORE: <number between 0 and 1>
REASON: <one short sentence>
""".strip()

_CONTEXTUAL_RELEVANCY_PROMPT = """
You are evaluating contextual relevancy for a RAG system.

Decide how relevant the retrieved context is for answering the user's question.
Return a score between 0 and 1.

Return exactly in this format:
SCORE: <number between 0 and 1>
REASON: <one short sentence>
""".strip()


@dataclass(slots=True)
class _MetricAssessment:
    """Internal normalized metric response."""

    score: float
    reason: str


class DeepEvalEvaluator(Evaluator):
    """Evaluate generation outputs with DeepEval-style metrics.

    The evaluator expects:
    - ``result.answer`` as the generated answer
    - ``result.contexts`` as the retrieved documents used for generation
    - optionally ``result.query.metadata['expected_output']`` for correctness

    If no reference answer is provided, correctness is skipped while the other
    retrieval-grounded metrics still run.
    """

    def __init__(
        self,
        client: ChatLLMClient,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = 256,
        correctness_threshold: float = 0.7,
        faithfulness_threshold: float = 0.7,
        contextual_relevancy_threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.correctness_threshold = correctness_threshold
        self.faithfulness_threshold = faithfulness_threshold
        self.contextual_relevancy_threshold = contextual_relevancy_threshold

    def _format_contexts(self, result: GenerationResult) -> str:
        """Format retrieved contexts into a stable plain-text block."""

        if not result.contexts:
            return "NO_CONTEXT"

        formatted_contexts: list[str] = []
        for index, document in enumerate(result.contexts, start=1):
            formatted_contexts.append(
                f"<doc{index}>\n"
                f"doc_id: {document.doc_id}\n"
                f"content: {document.text}\n"
                f"</doc{index}>"
            )
        return "\n\n".join(formatted_contexts)

    def _parse_assessment(self, raw_text: str) -> _MetricAssessment:
        """Parse ``SCORE`` and ``REASON`` fields from model output.

        The parser is intentionally tolerant so the evaluator still works if the
        model returns minor extra text around the requested format.
        """

        score_match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", raw_text, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.0
        score = max(0.0, min(1.0, score))

        reason_match = re.search(r"REASON:\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else raw_text.strip()

        return _MetricAssessment(score=score, reason=reason)

    def _assess(self, *, system_prompt: str, user_prompt: str) -> _MetricAssessment:
        """Run one metric grading call through the shared LLM layer."""

        response = self.client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self._parse_assessment(response.text)

    def _evaluate_correctness(self, result: GenerationResult) -> _MetricAssessment | None:
        """Evaluate correctness if a reference answer is available."""

        expected_output = result.query.metadata.get("expected_output")
        if not isinstance(expected_output, str) or not expected_output.strip():
            return None

        return self._assess(
            system_prompt=_CORRECTNESS_PROMPT,
            user_prompt=(
                f"Question:\n{result.query.text}\n\n"
                f"Reference answer:\n{expected_output}\n\n"
                f"Generated answer:\n{result.answer}\n"
            ),
        )

    def _evaluate_faithfulness(self, result: GenerationResult) -> _MetricAssessment:
        """Evaluate whether the answer is supported by retrieved context."""

        return self._assess(
            system_prompt=_FAITHFULNESS_PROMPT,
            user_prompt=(
                f"Question:\n{result.query.text}\n\n"
                f"Retrieved context:\n{self._format_contexts(result)}\n\n"
                f"Generated answer:\n{result.answer}\n"
            ),
        )

    def _evaluate_contextual_relevancy(self, result: GenerationResult) -> _MetricAssessment:
        """Evaluate whether retrieved context is relevant to the query."""

        return self._assess(
            system_prompt=_CONTEXTUAL_RELEVANCY_PROMPT,
            user_prompt=(
                f"Question:\n{result.query.text}\n\n"
                f"Retrieved context:\n{self._format_contexts(result)}\n"
            ),
        )

    def evaluate(self, result: GenerationResult) -> EvaluationResult:
        """Run DeepEval-style metrics and return normalized scores."""

        metrics: dict[str, float] = {}
        metadata: dict[str, object] = {
            "evaluation_strategy": "deep_eval_style",
            "reasons": {},
            "thresholds": {
                "correctness": self.correctness_threshold,
                "faithfulness": self.faithfulness_threshold,
                "contextual_relevancy": self.contextual_relevancy_threshold,
            },
        }

        correctness = self._evaluate_correctness(result)
        if correctness is not None:
            metrics["correctness"] = correctness.score
            metadata["reasons"]["correctness"] = correctness.reason
            metadata["correctness_passed"] = correctness.score >= self.correctness_threshold
        else:
            metadata["correctness_skipped"] = True
            metadata["reasons"]["correctness"] = (
                "Skipped because query.metadata['expected_output'] was not provided."
            )

        faithfulness = self._evaluate_faithfulness(result)
        metrics["faithfulness"] = faithfulness.score
        metadata["reasons"]["faithfulness"] = faithfulness.reason
        metadata["faithfulness_passed"] = (
            faithfulness.score >= self.faithfulness_threshold
        )

        contextual_relevancy = self._evaluate_contextual_relevancy(result)
        metrics["contextual_relevancy"] = contextual_relevancy.score
        metadata["reasons"]["contextual_relevancy"] = contextual_relevancy.reason
        metadata["contextual_relevancy_passed"] = (
            contextual_relevancy.score >= self.contextual_relevancy_threshold
        )

        return EvaluationResult(
            query=result.query,
            metrics=metrics,
            metadata=metadata,
        )
