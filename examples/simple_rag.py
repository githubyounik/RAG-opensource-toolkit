"""End-to-end RAG pipeline: PDF → nvidia/llama-nemotron-embed → retrieval → GLM-4.5 generation."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from rag_toolkit.core.types import Query
from rag_toolkit.embeddings import EmbeddingIndexer, OpenRouterEmbedder
from rag_toolkit.indexing import DocumentProcessor, PDFLoader
from rag_toolkit.pipelines import RAGPipeline
from rag_toolkit.retrieval import EmbeddingRetriever
from rag_toolkit.generation import ZhipuGenerator


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python examples/simple_rag.py <pdf_path> \"<query>\"\n"
            "Env vars required: OPENROUTER_API_KEY, ZHIPU_API_KEY"
        )

    pdf_path = sys.argv[1]
    query_text = sys.argv[2]

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    zhipu_key = os.environ.get("ZHIPU_API_KEY")

    if not openrouter_key:
        raise SystemExit("Missing env var: OPENROUTER_API_KEY")
    if not zhipu_key:
        raise SystemExit("Missing env var: ZHIPU_API_KEY")

    print(f"Indexing {pdf_path} ...")
    loader = PDFLoader()
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    embedder = OpenRouterEmbedder(api_key=openrouter_key)
    indexer = EmbeddingIndexer(embedder)

    parsed = loader.load(pdf_path)
    documents = processor.process(parsed)
    index = indexer.build(documents)
    print(f"Indexed {len(index)} chunks.")

    pipeline = RAGPipeline(
        retriever=EmbeddingRetriever(index=index, embedder=embedder, top_k=2),
        generator=ZhipuGenerator(api_key=zhipu_key),
    )

    generation_result, _ = pipeline.run(Query(text=query_text))

    print("\n=== Answer ===")
    print(generation_result.answer)


if __name__ == "__main__":
    main()
