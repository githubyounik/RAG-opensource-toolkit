"""Keyword-overlap retriever for the basic RAG flow."""

from __future__ import annotations

import re

from rag_toolkit.core.types import Document, Query, RetrievalResult
from rag_toolkit.indexing.simple_index import InMemoryIndex
from rag_toolkit.retrieval.base import Retriever


class SimpleRetriever(Retriever):
    """Retrieve chunks by counting token overlap with the query.

    This intentionally avoids vector databases and embeddings so the
    example stays focused on the module boundaries of a minimal RAG flow.
    """

    def __init__(self, index: InMemoryIndex, top_k: int = 2) -> None:
        super().__init__()
        self.index = index
        self.top_k = top_k

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower()))

    def _score_document(self, query_tokens: set[str], document: Document) -> int:
        document_tokens = self._tokenize(document.text)
        return len(query_tokens & document_tokens)

    def retrieve(self, query: Query) -> RetrievalResult:
        """Return the top chunks with the highest token-overlap scores."""

        query_tokens = self._tokenize(query.text)
        ranked_documents: list[tuple[int, Document]] = []

        for document in self.index.documents:
            score = self._score_document(query_tokens, document)
            if score > 0:
                scored_document = Document(
                    doc_id=document.doc_id,
                    text=document.text,
                    metadata={**document.metadata, "score": score},
                )
                ranked_documents.append((score, scored_document))

        ranked_documents.sort(key=lambda item: item[0], reverse=True)
        top_documents = [document for _, document in ranked_documents[: self.top_k]]

        return RetrievalResult(
            query=query,
            documents=top_documents,
            metadata={"top_k": self.top_k},
        )

