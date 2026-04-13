from rag_toolkit.indexing import (
    DocumentProcessor,
    PropositionProcessor,
    create_text_processor_from_config,
)


def test_indexing_factory_creates_default_processor() -> None:
    processor = create_text_processor_from_config(
        {
            "strategy": "default",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        }
    )

    assert isinstance(processor, DocumentProcessor)


def test_indexing_factory_creates_proposition_processor() -> None:
    processor = create_text_processor_from_config(
        {
            "strategy": "proposition",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "proposition": {
                "model": "nvidia/nemotron-3-super-120b-a12b:free",
                "temperature": 0.0,
                "max_tokens": 512,
                "max_retries": 2,
                "retry_delay_seconds": 2.0,
            },
        },
        openrouter_api_key="test-key",
    )

    assert isinstance(processor, PropositionProcessor)
