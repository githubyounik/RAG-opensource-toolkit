"""Sparse BM25 retriever over in-memory document chunks."""

from __future__ import annotations

import math
import re
from collections import Counter

from rag_toolkit.core.types import Document, Query, RetrievalResult
from rag_toolkit.retrieval.base import Retriever

_TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str, *, lowercase: bool) -> list[str]:
    normalized = text.lower() if lowercase else text
    return _TOKEN_PATTERN.findall(normalized)


class BM25Retriever(Retriever):
    """Retrieve documents with the Okapi BM25 ranking function.

    This retriever keeps a lightweight in-memory sparse index derived directly
    from the indexed ``Document`` objects. It does not depend on embeddings or
    a vector index, which makes it a natural parallel retrieval path beside the
    existing dense retriever.
    """

    def __init__(
        self,
        documents: list[Document],
        *,
        top_k: int = 4,
        k1: float = 1.5,
        b: float = 0.75,
        lowercase: bool = True,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if k1 < 0:
            raise ValueError("k1 cannot be negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")

        self.documents = documents
        self.top_k = top_k
        self.k1 = k1
        self.b = b
        self.lowercase = lowercase

        self._doc_tokens: list[list[str]] = []
        self._term_frequencies: list[Counter[str]] = []
        self._document_frequencies: Counter[str] = Counter()
        self._document_lengths: list[int] = []

        for tokens in (
            _tokenize(document.text, lowercase=self.lowercase)
            for document in self.documents
        ):
            self._doc_tokens.append(tokens)
            term_frequencies = Counter(tokens)
            self._term_frequencies.append(term_frequencies)
            self._document_lengths.append(len(tokens))
            self._document_frequencies.update(term_frequencies.keys())

        total_length = sum(self._document_lengths)
        self._average_document_length = (
            total_length / len(self._document_lengths) if self._document_lengths else 0.0
        )

    def _idf(self, term: str) -> float:
        document_frequency = self._document_frequencies.get(term, 0)
        document_count = len(self.documents)
        if document_count == 0:
            return 0.0
        return math.log(1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))

    def _score_document(self, *, query_terms: list[str], document_index: int) -> float:
        if not query_terms:
            return 0.0

        term_frequencies = self._term_frequencies[document_index]
        document_length = self._document_lengths[document_index]
        average_length = self._average_document_length or 1.0

        score = 0.0
        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if frequency == 0:
                continue

            idf = self._idf(term)
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * document_length / average_length
            )
            score += idf * (frequency * (self.k1 + 1.0)) / denominator

        return score

    def retrieve(self, query: Query) -> RetrievalResult:
        """Rank indexed documents by BM25 score for the given query."""

        query_terms = _tokenize(query.text, lowercase=self.lowercase)
        scored: list[tuple[float, Document]] = []

        for document_index, document in enumerate(self.documents):
            score = self._score_document(
                query_terms=query_terms,
                document_index=document_index,
            )
            scored_document = Document(
                doc_id=document.doc_id,
                text=document.text,
                metadata={
                    **document.metadata,
                    "score": score,
                    "retrieval_strategy": "bm25",
                },
            )
            scored.append((score, scored_document))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_documents = [document for _, document in scored[: self.top_k]]

        return RetrievalResult(
            query=query,
            documents=top_documents,
            metadata={
                "top_k": self.top_k,
                "document_count": len(self.documents),
                "retrieval_strategy": "bm25",
                "bm25_k1": self.k1,
                "bm25_b": self.b,
            },
        )
