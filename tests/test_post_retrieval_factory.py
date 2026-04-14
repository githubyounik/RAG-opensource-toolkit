from rag_toolkit.core.types import Document
from rag_toolkit.post_retrieval import (
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
            "irrelevant_chunk_penalty": 0.2,
            "rank_decay": 0.08,
            "max_segment_length": 6,
            "overall_max_length": 12,
            "minimum_segment_value": 0.15,
        },
        documents=documents,
    )

    assert isinstance(post_retriever, RelevantSegmentExtractor)
