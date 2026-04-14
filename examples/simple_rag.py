"""End-to-end RAG pipeline for PDF files."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from rag_toolkit.core import load_pipeline_config
from rag_toolkit.core.types import Query
from rag_toolkit.embeddings import EmbeddingIndexer, create_embedder_from_config
from rag_toolkit.evaluation import create_evaluator_from_config
from rag_toolkit.generation import create_generator_from_config
from rag_toolkit.indexing import PDFLoader, create_text_processor_from_config
from rag_toolkit.pipelines import RAGPipeline
from rag_toolkit.post_retrieval import create_post_retriever_from_config
from rag_toolkit.pre_retrieval import create_pre_retriever_from_config
from rag_toolkit.retrieval import EmbeddingRetriever


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python examples/simple_rag.py <pdf_path> \"<query>\"\n"
            "Env vars required depend on the configured generation provider."
        )

    pdf_path = sys.argv[1]
    query_text = sys.argv[2]

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    zhipu_key = os.environ.get("ZHIPU_API_KEY")

    # Read chunking parameters from the shared YAML config so they can be
    # changed without editing Python code.
    config = load_pipeline_config()
    embedding_config = config["embeddings"]
    document_processing_config = config["indexing"]["document_processing"]
    pre_retrieval_config = config.get("pre_retrieval")
    post_retrieval_config = config.get("post_retrieval")
    evaluation_config = config.get("evaluation")
    generation_config = config["generation"]

    if not openrouter_key:
        raise SystemExit("Missing env var: OPENROUTER_API_KEY")

    post_retrieval_strategy = str((post_retrieval_config or {}).get("strategy", "")).lower()
    rse_enabled = bool((post_retrieval_config or {}).get("enabled", False)) and (
        post_retrieval_strategy == "relevant_segment_extraction"
    )

    print(f"Indexing {pdf_path} ...")
    loader = PDFLoader()
    processor = create_text_processor_from_config(
        document_processing_config,
        openrouter_api_key=openrouter_key,
        force_non_overlapping_default=rse_enabled,
    )
    embedder = create_embedder_from_config(
        embedding_config,
        openrouter_api_key=openrouter_key,
    )
    indexer = EmbeddingIndexer(embedder)

    parsed = loader.load(pdf_path)
    documents = processor.process(parsed)
    index = indexer.build(documents)
    print(f"Indexed {len(index)} chunks.")

    pipeline = RAGPipeline(
        pre_retriever=create_pre_retriever_from_config(
            pre_retrieval_config,
            openrouter_api_key=openrouter_key,
            zhipu_api_key=zhipu_key,
        ),
        retriever=EmbeddingRetriever(index=index, embedder=embedder, top_k=2),
        post_retriever=create_post_retriever_from_config(
            post_retrieval_config,
            documents=index.documents,
            openrouter_api_key=openrouter_key,
            zhipu_api_key=zhipu_key,
        ),
        generator=create_generator_from_config(
            generation_config,
            openrouter_api_key=openrouter_key,
            zhipu_api_key=zhipu_key,
        ),
        evaluator=create_evaluator_from_config(
            evaluation_config,
            openrouter_api_key=openrouter_key,
            zhipu_api_key=zhipu_key,
        ),
    )

    generation_result, _ = pipeline.run(Query(text=query_text))

    print("\n=== Answer ===")
    print(generation_result.answer)


if __name__ == "__main__":
    main()
