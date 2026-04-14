from unittest.mock import patch

from rag_toolkit.core.types import Document
from rag_toolkit.post_retrieval import (
    ContextualCompressor,
    LLMReranker,
    RelevantSegmentExtractor,
    create_post_retriever_from_config,
)


def test_post_retrieval_factory_creates_relevant_segment_extractor() -> None:
    documents = [
        Document(
            doc_id="doc-0",
            text="chunk 0",
            metadata={"source": "doc", "chunk_id": 0, "start": 0, "end": 10},
        )
    ]

    post_retriever = create_post_retriever_from_config(
        {
            "enabled": True,
            "strategy": "relevant_segment_extraction",
            "relevant_segment_extraction": {
                "irrelevant_chunk_penalty": 0.2,
                "rank_decay": 0.08,
                "max_segment_length": 6,
                "overall_max_length": 12,
                "minimum_segment_value": 0.15,
            },
        },
        documents=documents,
    )

    assert isinstance(post_retriever, RelevantSegmentExtractor)


def test_post_retrieval_factory_creates_contextual_compressor() -> None:
    documents = [
        Document(
            doc_id="doc-0",
            text="chunk 0",
            metadata={"source": "doc", "chunk_id": 0, "start": 0, "end": 10},
        )
    ]

    with patch(
        "rag_toolkit.post_retrieval.factory.create_chat_llm_client",
        return_value=object(),
    ):
        post_retriever = create_post_retriever_from_config(
            {
                "enabled": True,
                "strategy": "contextual_compression",
                "contextual_compression": {
                    "provider": "openrouter",
                    "model": "google/gemma-4-26b-a4b-it:free",
                    "temperature": 0.0,
                    "max_tokens": 256,
                },
            },
            documents=documents,
            openrouter_api_key="test-key",
        )

    assert isinstance(post_retriever, ContextualCompressor)


def test_post_retrieval_factory_creates_llm_reranker() -> None:
    documents = [
        Document(
            doc_id="doc-0",
            text="chunk 0",
            metadata={"source": "doc", "chunk_id": 0, "start": 0, "end": 10},
        )
    ]

    with patch(
        "rag_toolkit.post_retrieval.factory.create_chat_llm_client",
        return_value=object(),
    ):
        post_retriever = create_post_retriever_from_config(
            {
                "enabled": True,
                "strategy": "rerank_llm",
                "rerank_llm": {
                    "provider": "openrouter",
                    "model": "z-ai/glm-5.1",
                    "temperature": 0.0,
                    "max_tokens": 32,
                    "top_k": 3,
                },
            },
            documents=documents,
            openrouter_api_key="test-key",
        )

    assert isinstance(post_retriever, LLMReranker)
