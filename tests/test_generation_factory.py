from unittest.mock import patch

from rag_toolkit.generation import OpenRouterGenerator, create_generator_from_config


def test_generation_factory_can_create_openrouter_generator() -> None:
    generator = create_generator_from_config(
        {
            "provider": "openrouter",
            "model": "google/gemma-4-26b-a4b-it:free",
            "temperature": 0.3,
            "max_tokens": 256,
            "max_retries": 3,
            "retry_delay_seconds": 1.5,
        },
        openrouter_api_key="test-key",
    )

    assert isinstance(generator, OpenRouterGenerator)
    assert generator.model == "google/gemma-4-26b-a4b-it:free"
    assert generator.temperature == 0.3
    assert generator.max_tokens == 256
    assert generator.max_retries == 3
    assert generator.retry_delay_seconds == 1.5


def test_generation_factory_can_create_zhipu_generator() -> None:
    with patch("rag_toolkit.generation.factory.ZhipuGenerator") as mock_generator:
        mock_generator.return_value = object()
        generator = create_generator_from_config(
            {
                "provider": "zhipu",
                "model": "glm-4.7",
                "temperature": 0.6,
                "max_tokens": None,
            },
            zhipu_api_key="test-key",
        )

    assert generator is mock_generator.return_value
