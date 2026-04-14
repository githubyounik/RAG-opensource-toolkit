from rag_toolkit.core.types import Document, Query, RetrievalResult
from rag_toolkit.post_retrieval.relevant_segment_extractor import RelevantSegmentExtractor


def test_relevant_segment_extractor_reconstructs_contiguous_segment() -> None:
    all_documents = [
        Document(
            doc_id=f"doc-chunk-{index}",
            text=f"chunk {index} text",
            metadata={
                "source": "doc",
                "chunk_id": index,
                "start": index * 10,
                "end": (index + 1) * 10,
            },
        )
        for index in range(5)
    ]

    result = RetrievalResult(
        query=Query(text="test query"),
        documents=[
            Document(
                doc_id="doc-chunk-1",
                text="chunk 1 text",
                metadata={"source": "doc", "chunk_id": 1, "score": 0.85},
            ),
            Document(
                doc_id="doc-chunk-3",
                text="chunk 3 text",
                metadata={"source": "doc", "chunk_id": 3, "score": 0.82},
            ),
        ],
    )

    extractor = RelevantSegmentExtractor(
        all_documents,
        irrelevant_chunk_penalty=0.2,
        rank_decay=0.05,
        max_segment_length=4,
        overall_max_length=6,
        minimum_segment_value=0.1,
    )

    refined = extractor.process(result)

    assert len(refined.documents) == 1
    segment = refined.documents[0]
    assert segment.metadata["segment_start_chunk_id"] == 1
    assert segment.metadata["segment_end_chunk_id"] == 3
    assert segment.metadata["covered_chunk_ids"] == [1, 2, 3]
    assert "chunk 1 text" in segment.text
    assert "chunk 2 text" in segment.text
    assert "chunk 3 text" in segment.text
