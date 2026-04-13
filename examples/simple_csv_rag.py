"""End-to-end RAG pipeline for CSV files.

This example intentionally reuses the same modular pipeline already used for
PDF files:

CSVLoader -> DocumentProcessor -> EmbeddingIndexer -> EmbeddingRetriever
-> configured Generator -> RAGPipeline

The only file-type-specific component here is the loader.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from rag_toolkit.core import load_pipeline_config
from rag_toolkit.core.types import Query
from rag_toolkit.embeddings import EmbeddingIndexer, create_embedder_from_config
from rag_toolkit.generation import create_generator_from_config
from rag_toolkit.indexing import CSVLoader, create_text_processor_from_config
from rag_toolkit.pipelines import RAGPipeline
from rag_toolkit.pre_retrieval import create_pre_retriever_from_config
from rag_toolkit.retrieval import EmbeddingRetriever


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python examples/simple_csv_rag.py <csv_path> \"<query>\"\n"
            "Env vars required depend on the configured generation provider."
        )

    csv_path = sys.argv[1]
    query_text = sys.argv[2]

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    zhipu_key = os.environ.get("ZHIPU_API_KEY")

    # Reuse the same config file as the PDF example so both file types are
    # controlled by one shared chunking configuration.
    config = load_pipeline_config()
    embedding_config = config["embeddings"]
    document_processing_config = config["indexing"]["document_processing"]
    pre_retrieval_config = config.get("pre_retrieval")
    generation_config = config["generation"]

    if not openrouter_key:
        raise SystemExit("Missing env var: OPENROUTER_API_KEY")

    print(f"Indexing {csv_path} ...")

    # Only the loader changes for CSV input. The rest of the pipeline stays
    # identical to the existing PDF RAG flow.
    loader = CSVLoader()
    processor = create_text_processor_from_config(
        document_processing_config,
        openrouter_api_key=openrouter_key,
    )
    embedder = create_embedder_from_config(
        embedding_config,
        openrouter_api_key=openrouter_key,
    )
    indexer = EmbeddingIndexer(embedder)

    parsed = loader.load(csv_path)
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
