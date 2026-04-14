"""Relevant Segment Extraction (RSE) post-retrieval component.

This module implements a lightweight version of the relevant segment extraction
idea from the reference notebook. After retrieval, it reconstructs contiguous
document segments by combining:

1. Absolute retrieval score
2. Retrieval rank
3. Original chunk order within the source document

The goal is to preserve neighboring context and fill gaps between highly
relevant chunks, instead of passing isolated chunks directly to the generator.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_toolkit.core.types import Document, RetrievalResult
from rag_toolkit.post_retrieval.base import PostRetriever


@dataclass(slots=True)
class _SegmentCandidate:
    """Internal segment candidate used by the optimizer."""

    source: str
    start_index: int
    end_index: int
    score: float


class RelevantSegmentExtractor(PostRetriever):
    """Reconstruct contiguous text segments from retrieved chunk clusters.

    Parameters
    ----------
    documents:
        Full list of indexed documents. This acts like the chunk text key-value
        store described in the reference notebook, letting us recover chunks
        that were not directly retrieved but sit between relevant ones.
    irrelevant_chunk_penalty:
        Constant penalty subtracted from each chunk value. Unretrieved chunks
        therefore become negative, while strongly retrieved chunks remain
        positive.
    rank_decay:
        Rank-based bonus. Higher-ranked retrieved chunks receive larger values,
        which helps stabilize the segment search when raw similarity scores are
        close together.
    max_segment_length:
        Maximum number of chunks allowed in a single reconstructed segment.
    overall_max_length:
        Maximum total number of chunks returned across all selected segments.
    minimum_segment_value:
        Minimum segment score required to keep a candidate segment.
    """

    def __init__(
        self,
        documents: list[Document],
        *,
        irrelevant_chunk_penalty: float = 0.2,
        rank_decay: float = 0.08,
        max_segment_length: int = 6,
        overall_max_length: int = 12,
        minimum_segment_value: float = 0.15,
    ) -> None:
        super().__init__()
        self.irrelevant_chunk_penalty = irrelevant_chunk_penalty
        self.rank_decay = rank_decay
        self.max_segment_length = max_segment_length
        self.overall_max_length = overall_max_length
        self.minimum_segment_value = minimum_segment_value

        # Build a simple chunk store keyed by source and chunk index.
        self._documents_by_source: dict[str, dict[int, Document]] = {}
        for document in documents:
            source = str(document.metadata.get("source", ""))
            chunk_id = int(document.metadata.get("chunk_id", -1))
            if not source or chunk_id < 0:
                continue
            self._documents_by_source.setdefault(source, {})[chunk_id] = document

    def _score_retrieved_documents(self, result: RetrievalResult) -> dict[str, dict[int, float]]:
        """Convert retrieved documents into per-source chunk relevance values."""

        scored_by_source: dict[str, dict[int, float]] = {}
        total_documents = max(len(result.documents), 1)

        for rank, document in enumerate(result.documents):
            source = str(document.metadata.get("source", ""))
            chunk_id = int(document.metadata.get("chunk_id", -1))
            raw_score = float(document.metadata.get("score", 0.0))

            if not source or chunk_id < 0:
                continue

            # Combine absolute score and a small rank bonus. This mirrors the
            # notebook's idea of mixing score and rank while staying simple.
            rank_bonus = self.rank_decay * (total_documents - rank)
            chunk_value = raw_score + rank_bonus
            scored_by_source.setdefault(source, {})[chunk_id] = chunk_value

        return scored_by_source

    def _segment_score(
        self,
        *,
        source: str,
        start_index: int,
        end_index: int,
        retrieved_scores: dict[int, float],
    ) -> float:
        """Calculate the score of one candidate segment."""

        score = 0.0
        for chunk_index in range(start_index, end_index):
            if chunk_index in retrieved_scores:
                score += retrieved_scores[chunk_index] - self.irrelevant_chunk_penalty
            else:
                score -= self.irrelevant_chunk_penalty
        return score

    def _find_best_segments(
        self,
        *,
        source: str,
        retrieved_scores: dict[int, float],
    ) -> list[_SegmentCandidate]:
        """Find the best-scoring contiguous segments for one source."""

        all_chunk_ids = sorted(self._documents_by_source.get(source, {}).keys())
        if not all_chunk_ids:
            return []

        max_chunk_index = all_chunk_ids[-1]
        selected_segments: list[_SegmentCandidate] = []
        used_chunk_indexes: set[int] = set()
        total_length = 0

        while total_length < self.overall_max_length:
            best_candidate: _SegmentCandidate | None = None

            for start_index in range(max_chunk_index + 1):
                if start_index in used_chunk_indexes:
                    continue

                for end_index in range(
                    start_index + 1,
                    min(start_index + self.max_segment_length + 1, max_chunk_index + 2),
                ):
                    candidate_indexes = set(range(start_index, end_index))
                    if candidate_indexes & used_chunk_indexes:
                        continue

                    score = self._segment_score(
                        source=source,
                        start_index=start_index,
                        end_index=end_index,
                        retrieved_scores=retrieved_scores,
                    )
                    if score < self.minimum_segment_value:
                        continue

                    candidate = _SegmentCandidate(
                        source=source,
                        start_index=start_index,
                        end_index=end_index,
                        score=score,
                    )
                    if best_candidate is None or candidate.score > best_candidate.score:
                        best_candidate = candidate

            if best_candidate is None:
                break

            candidate_length = best_candidate.end_index - best_candidate.start_index
            if total_length + candidate_length > self.overall_max_length:
                break

            selected_segments.append(best_candidate)
            used_chunk_indexes.update(
                range(best_candidate.start_index, best_candidate.end_index)
            )
            total_length += candidate_length

        return selected_segments

    def _build_segment_document(self, segment: _SegmentCandidate) -> Document | None:
        """Merge contiguous chunk texts into one larger segment document."""

        source_documents = self._documents_by_source.get(segment.source, {})
        merged_documents: list[Document] = []

        for chunk_index in range(segment.start_index, segment.end_index):
            document = source_documents.get(chunk_index)
            if document is not None:
                merged_documents.append(document)

        if not merged_documents:
            return None

        merged_text = "\n".join(document.text for document in merged_documents)
        first_document = merged_documents[0]
        last_document = merged_documents[-1]

        return Document(
            doc_id=(
                f"{segment.source}-segment-{segment.start_index}-{segment.end_index - 1}"
            ),
            text=merged_text,
            metadata={
                "source": segment.source,
                "segment_start_chunk_id": segment.start_index,
                "segment_end_chunk_id": segment.end_index - 1,
                "segment_score": segment.score,
                "chunk_count": len(merged_documents),
                "covered_chunk_ids": [
                    int(document.metadata["chunk_id"]) for document in merged_documents
                ],
                "post_retrieval_strategy": "relevant_segment_extraction",
                "original_chunk_doc_ids": [document.doc_id for document in merged_documents],
                "start": int(first_document.metadata.get("start", 0)),
                "end": int(last_document.metadata.get("end", 0)),
            },
        )

    def process(self, result: RetrievalResult) -> RetrievalResult:
        """Replace isolated retrieved chunks with contiguous reconstructed segments."""

        retrieved_scores_by_source = self._score_retrieved_documents(result)
        segment_documents: list[Document] = []

        for source, retrieved_scores in retrieved_scores_by_source.items():
            candidates = self._find_best_segments(
                source=source,
                retrieved_scores=retrieved_scores,
            )
            for candidate in candidates:
                segment_document = self._build_segment_document(candidate)
                if segment_document is not None:
                    segment_documents.append(segment_document)

        # If segment extraction cannot build anything useful, fall back to the
        # original retrieved chunks so the pipeline keeps working.
        if not segment_documents:
            return result

        segment_documents.sort(
            key=lambda document: float(document.metadata.get("segment_score", 0.0)),
            reverse=True,
        )

        return RetrievalResult(
            query=result.query,
            documents=segment_documents,
            metadata={
                **result.metadata,
                "post_retrieval_strategy": "relevant_segment_extraction",
                "original_retrieved_count": len(result.documents),
                "segment_count": len(segment_documents),
            },
        )
