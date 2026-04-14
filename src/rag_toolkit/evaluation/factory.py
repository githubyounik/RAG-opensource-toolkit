"""Factory helpers for evaluation components."""

from __future__ import annotations

from typing import Any

from rag_toolkit.evaluation.base import Evaluator
from rag_toolkit.evaluation.deep_eval_evaluator import DeepEvalEvaluator
from rag_toolkit.llm import create_chat_llm_client


def create_evaluator_from_config(
    evaluation_config: dict[str, Any] | None,
    *,
    openrouter_api_key: str | None = None,
    zhipu_api_key: str | None = None,
) -> Evaluator | None:
    """Create an evaluator from config.

    The first implementation follows the DeepEval-style notebook and evaluates:
    correctness, faithfulness, and contextual relevancy.
    """

    if not evaluation_config or not bool(evaluation_config.get("enabled", False)):
        return None

    strategy = str(evaluation_config.get("strategy", "deep_eval_style")).lower()
    if strategy != "deep_eval_style":
        raise ValueError(f"Unsupported evaluation strategy: {strategy}")

    llm_client = create_chat_llm_client(
        evaluation_config,
        openrouter_api_key=openrouter_api_key,
        zhipu_api_key=zhipu_api_key,
    )
    return DeepEvalEvaluator(
        llm_client,
        temperature=float(evaluation_config.get("temperature", 0.0)),
        max_tokens=(
            int(evaluation_config["max_tokens"])
            if evaluation_config.get("max_tokens") is not None
            else 256
        ),
        correctness_threshold=float(evaluation_config.get("correctness_threshold", 0.7)),
        faithfulness_threshold=float(evaluation_config.get("faithfulness_threshold", 0.7)),
        contextual_relevancy_threshold=float(
            evaluation_config.get("contextual_relevancy_threshold", 0.7)
        ),
    )
