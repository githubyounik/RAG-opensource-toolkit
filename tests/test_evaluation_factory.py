from __future__ import annotations

from unittest.mock import patch

from rag_toolkit.evaluation import DeepEvalEvaluator, create_evaluator_from_config


def test_evaluation_factory_creates_deep_eval_evaluator() -> None:
    with patch(
        "rag_toolkit.evaluation.factory.create_chat_llm_client",
        return_value=object(),
    ):
        evaluator = create_evaluator_from_config(
            {
                "enabled": True,
                "strategy": "deep_eval_style",
                "provider": "openrouter",
                "model": "z-ai/glm-5.1",
                "temperature": 0.0,
                "max_tokens": 256,
                "correctness_threshold": 0.7,
                "faithfulness_threshold": 0.7,
                "contextual_relevancy_threshold": 0.7,
            },
            openrouter_api_key="test-key",
        )

    assert isinstance(evaluator, DeepEvalEvaluator)
