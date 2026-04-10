"""Run the most basic end-to-end RAG flow."""

from __future__ import annotations

import sys

from rag_toolkit.core.types import Query
from rag_toolkit.evaluation import SimpleEvaluator
from rag_toolkit.generation import SimpleGenerator
from rag_toolkit.indexing import SimplePDFIndexer
from rag_toolkit.pipelines import RAGPipeline
from rag_toolkit.post_retrieval import SimplePostRetriever
from rag_toolkit.pre_retrieval import SimplePreRetriever
from rag_toolkit.retrieval import SimpleRetriever


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: PYTHONPATH=src python examples/simple_rag.py <pdf_path> <query>")

    pdf_path = sys.argv[1]
    query_text = sys.argv[2]

    # Index the PDF into in-memory chunks.
    indexer = SimplePDFIndexer(chunk_size=1000, chunk_overlap=200)
    index = indexer.index_pdf(pdf_path)

    # Build the pipeline from the module-specific components.
    pipeline = RAGPipeline(
        pre_retriever=SimplePreRetriever(),
        retriever=SimpleRetriever(index=index, top_k=2),
        post_retriever=SimplePostRetriever(),
        generator=SimpleGenerator(),
        evaluator=SimpleEvaluator(),
    )

    generation_result, evaluation_result = pipeline.run(Query(text=query_text))

    print("=== Answer ===")
    print(generation_result.answer)
    print()
    print("=== Evaluation ===")
    print(evaluation_result.metrics if evaluation_result else {})


if __name__ == "__main__":
    main()

