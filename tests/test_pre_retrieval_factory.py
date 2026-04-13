from unittest.mock import patch

from rag_toolkit.pre_retrieval import (
    QueryRewritePreRetriever,
    StepBackPreRetriever,
    create_pre_retriever_from_config,
)


def test_pre_retrieval_factory_creates_rewrite_component() -> None:
    pre_retriever = create_pre_retriever_from_config(
        {
            "enabled": True,
            "strategy": "rewrite",
            "provider": "openrouter",
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "temperature": 0.0,
            "max_tokens": 256,
            "max_retries": 2,
            "retry_delay_seconds": 2.0,
        },
        openrouter_api_key="test-key",
    )

    assert isinstance(pre_retriever, QueryRewritePreRetriever)


def test_pre_retrieval_factory_creates_step_back_component() -> None:
    pre_retriever = create_pre_retriever_from_config(
        {
            "enabled": True,
            "strategy": "step_back",
            "provider": "openrouter",
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "temperature": 0.0,
            "max_tokens": 256,
            "max_retries": 2,
            "retry_delay_seconds": 2.0,
        },
        openrouter_api_key="test-key",
    )

    assert isinstance(pre_retriever, StepBackPreRetriever)


def test_pre_retrieval_factory_returns_none_when_disabled() -> None:
    pre_retriever = create_pre_retriever_from_config(
        {
            "enabled": False,
            "strategy": "rewrite",
        },
        openrouter_api_key="test-key",
    )

    assert pre_retriever is None


def test_pre_retrieval_factory_supports_zhipu() -> None:
    with patch(
        "rag_toolkit.pre_retrieval.factory.create_chat_llm_client",
        return_value=object(),
    ):
        pre_retriever = create_pre_retriever_from_config(
            {
                "enabled": True,
                "strategy": "rewrite",
                "provider": "zhipu",
                "model": "glm-4.7",
                "temperature": 0.0,
                "max_tokens": 256,
            },
            zhipu_api_key="test-key",
        )

    assert isinstance(pre_retriever, QueryRewritePreRetriever)
