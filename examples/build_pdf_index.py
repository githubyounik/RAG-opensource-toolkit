"""Pre-build and persist a FAISS index for one PDF file."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rag_toolkit.core import load_pipeline_config
from rag_toolkit.embeddings import EmbeddingIndexer, create_embedder_from_config
from rag_toolkit.indexing import PDFLoader, create_text_processor_from_config


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python examples/build_pdf_index.py <pdf_path>\n"
            "This builds and saves a local FAISS index for later retrieval reuse."
        )

    pdf_path = sys.argv[1]
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise SystemExit("Missing env var: OPENROUTER_API_KEY")

    config = load_pipeline_config()
    embedding_config = config["embeddings"]
    document_processing_config = config["indexing"]["document_processing"]
    storage_config = embedding_config.get("storage", {})
    cache_dir = str(storage_config.get("cache_dir", ".rag_cache/faiss"))

    if not bool(storage_config.get("enabled", False)):
        raise SystemExit(
            "embeddings.storage.enabled must be true in the config to build a persisted FAISS index."
        )

    embedder = create_embedder_from_config(
        embedding_config,
        openrouter_api_key=openrouter_key,
    )
    indexer = EmbeddingIndexer(embedder)
    cached_index_dir = indexer.lookup_cached_index_for_file(
        pdf_path,
        cache_dir=cache_dir,
    )
    if cached_index_dir is not None:
        from rag_toolkit.embeddings import VectorIndex

        index = VectorIndex.load(cached_index_dir)
        print(f"Loaded existing FAISS index with {len(index)} chunks.")
        print(f"Index directory: {cached_index_dir}")
        return

    print(f"Preparing chunks for {pdf_path} ...")
    loader = PDFLoader()
    processor = create_text_processor_from_config(
        document_processing_config,
        openrouter_api_key=openrouter_key,
    )
    parsed = loader.load(pdf_path)
    documents = processor.process(parsed)

    index, index_dir, loaded_from_disk = indexer.build_or_load(
        documents,
        cache_dir=cache_dir,
        reuse_existing=bool(storage_config.get("reuse_existing", True)),
        namespace=Path(pdf_path).stem,
    )
    indexer.register_cached_index_for_file(
        pdf_path,
        cache_dir=cache_dir,
        index_directory=index_dir,
    )

    action = "Loaded existing" if loaded_from_disk else "Built new"
    print(f"{action} FAISS index with {len(index)} chunks.")
    print(f"Index directory: {index_dir}")


if __name__ == "__main__":
    main()
