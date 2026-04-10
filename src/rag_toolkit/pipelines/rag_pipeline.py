"""Composable RAG pipeline skeleton."""

from __future__ import annotations

from rag_toolkit.core.types import EvaluationResult, GenerationResult, Query
from rag_toolkit.evaluation.base import Evaluator
from rag_toolkit.generation.base import Generator
from rag_toolkit.post_retrieval.base import PostRetriever
from rag_toolkit.pre_retrieval.base import PreRetriever
from rag_toolkit.retrieval.base import Retriever


class RAGPipeline:
    """Orchestrates modular RAG components."""

    def __init__(
        self,
        *,
        pre_retriever: PreRetriever | None = None,
        retriever: Retriever,
        post_retriever: PostRetriever | None = None,
        generator: Generator,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.pre_retriever = pre_retriever
        self.retriever = retriever
        self.post_retriever = post_retriever
        self.generator = generator
        self.evaluator = evaluator

    def run(self, query: Query) -> tuple[GenerationResult, EvaluationResult | None]:
        """Run the end-to-end RAG flow using configured modules."""
        processed_query = self.pre_retriever.process(query) if self.pre_retriever else query
        retrieval_result = self.retriever.retrieve(processed_query)
        refined_result = (
            self.post_retriever.process(retrieval_result)
            if self.post_retriever
            else retrieval_result
        )
        generation_result = self.generator.generate(refined_result)
        evaluation_result = (
            self.evaluator.evaluate(generation_result) if self.evaluator else None
        )
        return generation_result, evaluation_result
