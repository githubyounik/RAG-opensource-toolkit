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
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_toolkit.core import load_pipeline_config
from rag_toolkit.core.types import Query
from rag_toolkit.embeddings import EmbeddingIndexer, create_embedder_from_config
from rag_toolkit.evaluation import create_evaluator_from_config
from rag_toolkit.generation import create_generator_from_config
from rag_toolkit.indexing import CSVLoader, create_text_processor_from_config
from rag_toolkit.pipelines import RAGPipeline
from rag_toolkit.post_retrieval import create_post_retriever_from_config
from rag_toolkit.pre_retrieval import create_pre_retriever_from_config
from rag_toolkit.retrieval import create_retriever_from_config
from rag_toolkit.run_logger import save_run_log


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
    retrieval_config = config.get("retrieval")
    post_retrieval_config = config.get("post_retrieval")
    evaluation_config = config.get("evaluation")
    generation_config = config["generation"]

    retrieval_strategy = str((retrieval_config or {}).get("strategy", "embedding")).lower()
    embedding_required = retrieval_strategy in {"embedding", "hybrid"}
    storage_config = embedding_config.get("storage", {})
    storage_enabled = embedding_required and bool(storage_config.get("enabled", False))

    if embedding_required and not openrouter_key:
        raise SystemExit("Missing env var: OPENROUTER_API_KEY")

    post_retrieval_strategy = str((post_retrieval_config or {}).get("strategy", "")).lower()
    rse_enabled = bool((post_retrieval_config or {}).get("enabled", False)) and (
        post_retrieval_strategy == "relevant_segment_extraction"
    )

    print(f"Indexing {csv_path} ...")

    # Only the loader changes for CSV input. The rest of the pipeline stays
    # identical to the existing PDF RAG flow.
    loader = CSVLoader()
    processor = create_text_processor_from_config(
        document_processing_config,
        openrouter_api_key=openrouter_key,
        force_non_overlapping_default=rse_enabled,
    )
    parsed = loader.load(csv_path)
    documents = processor.process(parsed)
    embedder = None
    index = None
    if embedding_required:
        embedder = create_embedder_from_config(
            embedding_config,
            openrouter_api_key=openrouter_key,
        )
        indexer = EmbeddingIndexer(embedder)
        if storage_enabled:
            index, index_dir, loaded_from_disk = indexer.build_or_load(
                documents,
                cache_dir=str(storage_config.get("cache_dir", ".rag_cache/faiss")),
                reuse_existing=bool(storage_config.get("reuse_existing", True)),
                namespace=Path(csv_path).stem,
            )
            action = "Loaded" if loaded_from_disk else "Built"
            print(f"{action} FAISS index with {len(index)} chunks: {index_dir}")
        else:
            index = indexer.build(documents)
            print(f"Indexed {len(index)} chunks.")
    else:
        print(f"Prepared {len(documents)} chunks for BM25 retrieval.")

    pipeline = RAGPipeline(
        pre_retriever=create_pre_retriever_from_config(
            pre_retrieval_config,
            openrouter_api_key=openrouter_key,
            zhipu_api_key=zhipu_key,
        ),
        retriever=create_retriever_from_config(
            retrieval_config,
            documents=documents,
            index=index,
            embedder=embedder,
        ),
        post_retriever=create_post_retriever_from_config(
            post_retrieval_config,
            documents=documents,
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

    generation_result, evaluation_result = pipeline.run(Query(text=query_text))

    print("\n=== Answer ===")
    print(generation_result.answer)

    log_file = save_run_log(
        config=config,
        query_text=query_text,
        generation_result=generation_result,
        evaluation_result=evaluation_result,
    )
    print(f"\n=== Run log saved to: {log_file} ===")


if __name__ == "__main__":
    main()
