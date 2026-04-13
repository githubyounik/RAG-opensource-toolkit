"""End-to-end RAG pipeline for PDF files."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from rag_toolkit.core import load_pipeline_config
from rag_toolkit.core.types import Query
from rag_toolkit.embeddings import EmbeddingIndexer, OpenRouterEmbedder
from rag_toolkit.generation import create_generator_from_config
from rag_toolkit.indexing import PDFLoader, create_text_processor_from_config
from rag_toolkit.pipelines import RAGPipeline
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
    if not openrouter_key:
        raise SystemExit("Missing env var: OPENROUTER_API_KEY")
    zhipu_key = os.environ.get("ZHIPU_API_KEY")

    # Read chunking parameters from the shared YAML config so they can be
    # changed without editing Python code.
    config = load_pipeline_config()
    document_processing_config = config["indexing"]["document_processing"]
    generation_config = config["generation"]
    chunk_size = int(document_processing_config["chunk_size"])
    chunk_overlap = int(document_processing_config["chunk_overlap"])

    print(f"Indexing {pdf_path} ...")
    loader = PDFLoader()
    processor = create_text_processor_from_config(
        document_processing_config,
        openrouter_api_key=openrouter_key,
    )
    embedder = OpenRouterEmbedder(api_key=openrouter_key)
    indexer = EmbeddingIndexer(embedder)

    parsed = loader.load(pdf_path)
    documents = processor.process(parsed)
    index = indexer.build(documents)
    print(f"Indexed {len(index)} chunks.")

    pipeline = RAGPipeline(
        retriever=EmbeddingRetriever(index=index, embedder=embedder, top_k=2),
        generator=create_generator_from_config(
            generation_config,
            openrouter_api_key=openrouter_key,
            zhipu_api_key=zhipu_key,
        ),
    )

    generation_result, _ = pipeline.run(Query(text=query_text))

    print("\n=== Answer ===")
    print(generation_result.answer)


if __name__ == "__main__":
    main()
