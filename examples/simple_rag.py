"""End-to-end RAG pipeline for PDF files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_toolkit.core import load_pipeline_config
from rag_toolkit.core.types import Query
from rag_toolkit.embeddings import EmbeddingIndexer, VectorIndex, create_embedder_from_config
from rag_toolkit.evaluation import create_evaluator_from_config
from rag_toolkit.generation import create_generator_from_config
from rag_toolkit.indexing import PDFLoader, create_text_processor_from_config
from rag_toolkit.pipelines import RAGPipeline
from rag_toolkit.post_retrieval import create_post_retriever_from_config
from rag_toolkit.pre_retrieval import create_pre_retriever_from_config
from rag_toolkit.retrieval import create_retriever_from_config
from rag_toolkit.run_logger import save_run_log


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit(
            "Usage: python examples/simple_rag.py [<pdf_path>] \"<query>\"\n"
            "Env vars required depend on the configured generation provider."
        )

    pdf_path = sys.argv[1] if len(sys.argv) == 3 else None
    query_text = sys.argv[2] if len(sys.argv) == 3 else sys.argv[1]

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    zhipu_key = os.environ.get("ZHIPU_API_KEY")

    # Read chunking parameters from the shared YAML config so they can be
    # changed without editing Python code.
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

    documents = []
    embedder = None
    index = None
    if pdf_path is not None:
        print(f"Indexing {pdf_path} ...")
        loader = PDFLoader()
        processor = create_text_processor_from_config(
            document_processing_config,
            openrouter_api_key=openrouter_key,
            force_non_overlapping_default=rse_enabled,
        )
        parsed = loader.load(pdf_path)
        documents = processor.process(parsed)

    if embedding_required:
        embedder = create_embedder_from_config(
            embedding_config,
            openrouter_api_key=openrouter_key,
        )
        indexer = EmbeddingIndexer(embedder)
        cache_dir = str(storage_config.get("cache_dir", ".rag_cache/faiss"))
        if pdf_path is None:
            if not storage_enabled:
                raise SystemExit(
                    "Querying all indexed content without a file requires embeddings.storage.enabled=true."
                )
            index = VectorIndex.load_all(cache_dir)
            documents = index.documents
            print(f"Loaded merged FAISS index with {len(index)} chunks from: {cache_dir}")
        else:
            if storage_enabled:
                index, index_dir, loaded_from_disk = indexer.build_or_load(
                    documents,
                    cache_dir=cache_dir,
                    reuse_existing=bool(storage_config.get("reuse_existing", True)),
                    namespace=Path(pdf_path).stem,
                )
                action = "Loaded" if loaded_from_disk else "Built"
                print(f"{action} FAISS index with {len(index)} chunks: {index_dir}")
            else:
                index = indexer.build(documents)
                print(f"Indexed {len(index)} chunks.")
    else:
        if pdf_path is None:
            raise SystemExit("BM25-only search requires an input file to provide searchable chunks.")
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
