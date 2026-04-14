from __future__ import annotations

from rag_toolkit.core.types import Document, GenerationResult, Query
from rag_toolkit.evaluation.deep_eval_evaluator import DeepEvalEvaluator


class DummyClient:
    """Simple fake LLM client that returns queued rubric outputs."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
    ):
        class Response:
            def __init__(self, text: str) -> None:
                self.text = text

        return Response(self.outputs.pop(0))


def test_deep_eval_evaluator_scores_three_metrics() -> None:
    evaluator = DeepEvalEvaluator(
        DummyClient(
            [
                "SCORE: 0.9\nREASON: The answer matches the expected answer.",
                "SCORE: 0.8\nREASON: The answer is supported by the context.",
                "SCORE: 0.7\nREASON: The retrieved context is relevant to the question.",
            ]
        )
    )

    result = GenerationResult(
        query=Query(
            text="What is the capital of Spain?",
            metadata={"expected_output": "Madrid is the capital of Spain."},
        ),
        answer="Madrid is the capital of Spain.",
        contexts=[Document(doc_id="doc-1", text="Madrid is the capital of Spain.")],
    )

    evaluation = evaluator.evaluate(result)

    assert evaluation.metrics["correctness"] == 0.9
    assert evaluation.metrics["faithfulness"] == 0.8
    assert evaluation.metrics["contextual_relevancy"] == 0.7
    assert evaluation.metadata["correctness_passed"] is True
    assert evaluation.metadata["faithfulness_passed"] is True
    assert evaluation.metadata["contextual_relevancy_passed"] is True


def test_deep_eval_evaluator_skips_correctness_without_reference_answer() -> None:
    evaluator = DeepEvalEvaluator(
        DummyClient(
            [
                "SCORE: 0.6\nREASON: Mostly grounded in the context.",
                "SCORE: 0.9\nREASON: The context is highly relevant.",
            ]
        )
    )

    result = GenerationResult(
        query=Query(text="What is 3+3?"),
        answer="6",
        contexts=[Document(doc_id="doc-1", text="6")],
    )

    evaluation = evaluator.evaluate(result)

    assert "correctness" not in evaluation.metrics
    assert evaluation.metadata["correctness_skipped"] is True
    assert evaluation.metrics["faithfulness"] == 0.6
    assert evaluation.metrics["contextual_relevancy"] == 0.9
