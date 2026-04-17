# RAG Open Source Toolkit

A modular Python toolkit for building Retrieval-Augmented Generation (RAG) pipelines.

This repository is no longer only a framework skeleton. It now contains a minimal end-to-end RAG flow built around:

- PDF loading
- CSV loading
- Text cleaning and chunking
- Pre-retrieval query transformation
- Embedding-based indexing
- Dense embedding retrieval
- Sparse BM25 retrieval
- Hybrid retrieval with RRF fusion
- LLM generation
- Pipeline orchestration

## Current Architecture

The current codebase supports both PDF and CSV input, and both follow the same high-level pipeline:

```text
FileLoader
  -> configured TextProcessor
  -> list[Document]
  -> optional configured PreRetriever
  -> configured Retriever
  -> configured Generator
  -> RAGPipeline
```

The retrieval layer now supports three paths:

```text
Dense path:
FileLoader
  -> configured TextProcessor
  -> list[Document]
  -> configured Embedder
  -> EmbeddingIndexer
  -> VectorIndex
  -> EmbeddingRetriever
  -> configured Generator
  -> RAGPipeline
```

```text
Sparse path:
FileLoader
  -> configured TextProcessor
  -> list[Document]
  -> BM25Retriever
  -> configured Generator
  -> RAGPipeline
```

```text
Hybrid path:
FileLoader
  -> configured TextProcessor
  -> list[Document]
  -> configured Embedder
  -> EmbeddingIndexer
  -> VectorIndex
  -> EmbeddingRetriever + BM25Retriever
  -> HybridRetriever (RRF fusion)
  -> configured Generator
  -> RAGPipeline
```

Concrete loaders currently implemented:

```text
PDFLoader
CSVLoader
```

Example flow for a PDF file:

```text
PDFLoader
  -> configured TextProcessor
  -> list[Document]
  -> optional configured PreRetriever
  -> configured Retriever
  -> configured Generator
  -> RAGPipeline
```

Example flow for a CSV file:

```text
CSVLoader
  -> configured TextProcessor
  -> list[Document]
  -> optional configured PreRetriever
  -> configured Retriever
  -> configured Generator
  -> RAGPipeline
```

In plain language, the pipeline does this:

1. Load a source file and extract structured text.
2. Clean and split the text into chunks.
3. Optionally rewrite or broaden the user query before retrieval.
4. Choose a retrieval strategy.
5. For dense retrieval, convert chunks into embeddings and store them in an in-memory vector index.
6. Retrieve relevant chunks with embedding similarity, BM25 keyword matching, or a hybrid of both.
7. Send the retrieved context to a generation model.
8. Return the final answer through a unified pipeline interface.

## Module Overview

### `core`

Shared base classes and data models used across the whole project.

- `ParsedFile`: unified output of file loading
- `Query`: input query object
- `Document`: chunk-level document object
- `RetrievalResult`: retrieval stage output
- `GenerationResult`: generation stage output
- `EvaluationResult`: evaluation stage output

### `indexing`

Responsible for file loading and turning raw files into document chunks.

- `PDFLoader`: reads PDF files and returns `ParsedFile`
- `CSVLoader`: reads CSV files and converts each row into readable text
- `DocumentProcessor`: regular fixed-size chunking
- `PropositionProcessor`: proposition-level chunking powered by an LLM
- `create_text_processor_from_config`: chooses the chunking strategy from the config file
- `FileLoader`, `TextProcessor`: base interfaces for indexing-related components

### `embeddings`

Responsible for dense vector creation and vector index construction.

- `OpenRouterEmbedder`: calls embedding models through OpenRouter
- `create_embedder_from_config`: chooses the embedding backend from the config file
- `EmbeddingIndexer`: embeds document chunks and builds a `VectorIndex`
- `VectorIndex`: in-memory storage for documents and their embeddings

### `retrieval`

Responsible for finding relevant chunks for a query.

- `EmbeddingRetriever`: embeds the query and ranks chunks by cosine similarity
- `BM25Retriever`: tokenizes indexed chunks and ranks them with sparse BM25 scoring
- `HybridRetriever`: runs both retrieval paths and fuses their rankings with Reciprocal Rank Fusion (RRF)
- `create_retriever_from_config`: chooses the retrieval strategy from the config file

### `generation`

Responsible for generating the final answer from retrieved context.

- `ZhipuGenerator`: sends retrieved context and query to a ZhipuAI GLM model
- `OpenRouterGenerator`: sends retrieved context and query to an OpenRouter chat model
- `create_generator_from_config`: chooses the generation backend from the config file

### `pre_retrieval`

Responsible for query transformation before retrieval.

- `QueryRewritePreRetriever`: rewrites the original question into a more retrieval-friendly query
- `StepBackPreRetriever`: rewrites the question into a broader background query
- `HyDEPreRetriever`: generates a hypothetical answer document and uses it as retrieval text
- `QueryTransformer`: reusable query transformation helper built on the shared LLM layer
- `create_pre_retriever_from_config`: chooses the pre-retrieval strategy from the config file

### `llm`

Shared provider layer reused by both `generation` and `pre_retrieval`.

- `OpenRouterChatClient`: shared OpenRouter chat client with retry and delay handling
- `ZhipuChatClient`: shared Zhipu chat client
- `create_chat_llm_client`: chooses the provider client from config

### `pipelines`

Responsible for chaining modules together.

- `RAGPipeline`: standard orchestration entry for pre-retrieval, retrieval, post-retrieval, generation, and evaluation

### `post_retrieval`, `evaluation`

`post_retrieval` now includes:

- `RelevantSegmentExtractor`: reconstructs contiguous document segments from nearby retrieved chunks
- `ContextualCompressor`: compresses each retrieved chunk down to only the query-relevant content
- `CohereReranker`: calls OpenRouter's rerank API with a dedicated rerank model such as `cohere/rerank-v3.5`
- `create_post_retriever_from_config`: chooses the post-retrieval strategy from the config file

`evaluation` now includes:

- `DeepEvalEvaluator`: DeepEval-style evaluation for correctness, faithfulness, and contextual relevancy
- `create_evaluator_from_config`: chooses the evaluation strategy from the config file

## Project Layout

```text
RAG-opensource-toolkit/
├── .env.example
├── examples/
├── pyproject.toml
├── README.md
├── configs/
├── docs/
├── src/
│   └── rag_toolkit/
│       ├── core/
│       ├── embeddings/
│       ├── evaluation/
│       ├── generation/
│       ├── indexing/
│       ├── pipelines/
│       ├── post_retrieval/
│       ├── pre_retrieval/
│       └── retrieval/
└── tests/
```

## Installation

Use Python 3.10+.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you want to run the current end-to-end example, install the optional LLM dependencies too:

```bash
pip install -e ".[dev,llm]"
```

## Environment Variables

Create a `.env` file based on `./.env.example`:

```env
OPENROUTER_API_KEY=your-openrouter-key-here
ZHIPU_API_KEY=your-zhipu-key-here
```

## Chunking Configuration

The chunking parameters are now configurable through [configs/pipeline.example.yaml](/home/test/Desktop/code/RAG-opensource-toolkit/configs/pipeline.example.yaml).

Current settings:

```yaml
indexing:
  document_processing:
    strategy: proposition
    chunk_size: 1000
    chunk_overlap: 200
    proposition:
      model: nvidia/nemotron-3-super-120b-a12b:free
      temperature: 0.0
      max_tokens: 512
      max_retries: 2
      retry_delay_seconds: 2.0
```

You can switch chunking strategies through `indexing.document_processing.strategy`:

```yaml
indexing:
  document_processing:
    strategy: default
```

or

```yaml
indexing:
  document_processing:
    strategy: proposition
```

`default` uses regular fixed-size chunking through `DocumentProcessor`.

`proposition` first creates regular base chunks and then rewrites each chunk into proposition-sized documents through `PropositionProcessor`.

## Embedding Configuration

Embedding provider and model are configured through [configs/pipeline.example.yaml](/home/test/Desktop/code/RAG-opensource-toolkit/configs/pipeline.example.yaml).

Current default:

```yaml
embeddings:
  provider: openrouter
  model: perplexity/pplx-embed-v1-0.6b
```

Right now the toolkit supports:

- `openrouter`

Embeddings are required when `retrieval.strategy: embedding` or `retrieval.strategy: hybrid`.

## Retrieval Configuration

Retrieval is configured through [configs/pipeline.example.yaml](/home/test/Desktop/code/RAG-opensource-toolkit/configs/pipeline.example.yaml).

Current example:

```yaml
retrieval:
  strategy: embedding
  embedding:
    top_k: 4
  bm25:
    top_k: 4
    k1: 1.5
    b: 0.75
    lowercase: true
  hybrid:
    top_k: 4
    rrf_k: 60
```

Supported `strategy` values:

- `embedding`: build an in-memory `VectorIndex`, embed the query, and retrieve by cosine similarity
- `bm25`: skip embeddings entirely and retrieve directly from chunk text with BM25
- `hybrid`: run both dense and sparse retrieval, then fuse the ranked lists with RRF

Key parameters:

- `embedding.top_k`: number of chunks returned by dense retrieval
- `bm25.top_k`: number of chunks returned by BM25
- `bm25.k1`: BM25 term-frequency saturation parameter
- `bm25.b`: BM25 document-length normalization parameter
- `bm25.lowercase`: whether document and query text are lowercased before tokenization
- `hybrid.top_k`: number of chunks kept after fusion
- `hybrid.rrf_k`: Reciprocal Rank Fusion constant used when combining the embedding and BM25 ranked lists

## Pre-Retrieval Configuration

Pre-retrieval is optional and is configured through [configs/pipeline.example.yaml](/home/test/Desktop/code/RAG-opensource-toolkit/configs/pipeline.example.yaml).

Current default:

```yaml
pre_retrieval:
  enabled: true
  strategy: rewrite
  provider: openrouter
  model: z-ai/glm-5.1
  temperature: 0.0
  max_tokens: 256
  max_retries: 2
  retry_delay_seconds: 2.0
  hyde_target_char_length: 800
```

Supported `strategy` values:

- `rewrite`: rewrite the original question into a more retrieval-friendly query
- `step_back`: rewrite the original question into a broader background query
- `hyde`: generate a hypothetical answer document first, then use that document as the retrieval query

Supported `provider` values:

- `openrouter`
- `zhipu`

`hyde_target_char_length` is only used when `strategy: hyde`. It provides a
soft length target for the hypothetical document so the generated text is
closer to the size of indexed chunks.

## Post-Retrieval Configuration

Post-retrieval is optional and is configured through [configs/pipeline.example.yaml](/home/test/Desktop/code/RAG-opensource-toolkit/configs/pipeline.example.yaml).

Current example:

```yaml
post_retrieval:
  enabled: false
  strategy: relevant_segment_extraction
  relevant_segment_extraction:
    irrelevant_chunk_penalty: 0.2
    rank_decay: 0.08
    max_segment_length: 6
    overall_max_length: 12
    minimum_segment_value: 0.15
  contextual_compression:
    provider: openrouter
    model: z-ai/glm-5.1
    temperature: 0.0
    max_tokens: 1200
    max_retries: 2
    retry_delay_seconds: 2.0
  rerank:
    provider: openrouter
    model: cohere/rerank-v3.5
    top_k: 3
    max_tokens_per_doc:
    max_retries: 2
    retry_delay_seconds: 2.0
```

Supported `strategy` values:

- `relevant_segment_extraction`: merge nearby relevant chunks into contiguous context segments
- `contextual_compression`: use an LLM to compress each retrieved chunk to only the query-relevant content
- `rerank`: use OpenRouter's dedicated rerank endpoint to reorder retrieved chunks with a rerank model

Key parameters:

- `relevant_segment_extraction.*`: parameters used only when `strategy: relevant_segment_extraction`
- `contextual_compression.*`: parameters used only when `strategy: contextual_compression`
- `rerank.*`: parameters used only when `strategy: rerank`
- `rerank.top_k`: maximum number of documents kept after reranking
- `rerank.max_tokens_per_doc`: optional per-document truncation budget passed to the rerank API
- `rerank.model`: the rerank model name sent to OpenRouter, for example `cohere/rerank-v3.5`

Important constraint:

- When `post_retrieval.enabled: true` and `strategy: relevant_segment_extraction`, the toolkit will automatically force indexing to use `DocumentProcessor` with `chunk_overlap = 0`
- This override happens in code even if the YAML file contains `strategy: proposition` or a non-zero overlap

This constraint exists because Relevant Segment Extraction depends on clean,
non-overlapping chunk boundaries so contiguous segments can be reconstructed
reliably.

For `contextual_compression` and `rerank`, there is no special chunking
override. They operate on the chunks returned by the retriever after retrieval.

## Evaluation Configuration

Evaluation is optional and is configured through [configs/pipeline.example.yaml](/home/test/Desktop/code/RAG-opensource-toolkit/configs/pipeline.example.yaml).

Current example:

```yaml
evaluation:
  enabled: false
  strategy: deep_eval_style
  provider: openrouter
  model: z-ai/glm-5.1
  temperature: 0.0
  max_tokens: 256
  correctness_threshold: 0.7
  faithfulness_threshold: 0.7
  contextual_relevancy_threshold: 0.7
  max_retries: 2
  retry_delay_seconds: 2.0
```

Supported `strategy` values:

- `deep_eval_style`: evaluate correctness, faithfulness, and contextual relevancy with the shared LLM layer

Important notes:

- `correctness` requires `query.metadata["expected_output"]`
- if no reference answer is provided, correctness is skipped automatically
- `faithfulness` uses the retrieved contexts attached to `GenerationResult.contexts`
- `contextual_relevancy` checks whether the retrieved contexts are useful for the query

## Generation Model Configuration

The generation provider and model are also controlled through [configs/pipeline.example.yaml](/home/test/Desktop/code/RAG-opensource-toolkit/configs/pipeline.example.yaml).

Current default:

```yaml
generation:
  provider: openrouter
  model: z-ai/glm-5.1
  temperature: 0.6
  max_tokens:
  max_retries: 2
  retry_delay_seconds: 2.0
```

If you want to use Zhipu generation instead, you can switch it to:

```yaml
generation:
  provider: zhipu
  model: glm-4.7
  temperature: 0.6
  max_tokens:
```

If you want to use a different OpenRouter model, you can directly put the full model id in the config:

```yaml
generation:
  provider: openrouter
  model: z-ai/glm-5.1
  temperature: 0.6
  max_tokens:
  max_retries: 2
  retry_delay_seconds: 2.0
```

The OpenRouter generator also includes retry logic with delay for transient failures such as `429 Too Many Requests`.

## Run The Examples

PDF example:

```bash
PYTHONPATH=src python examples/simple_rag.py <pdf_path> "<your question>"
```

Example:

```bash
PYTHONPATH=src python examples/simple_rag.py docs/Understanding_Climate_Change.pdf "What marked the beginning of the modern climate era and human civilization?"
```

CSV example:

```bash
PYTHONPATH=src python examples/simple_csv_rag.py <csv_path> "<your question>"
```

Example:

```bash
PYTHONPATH=src python examples/simple_csv_rag.py docs/customers-100.csv "how can i contact Sheryl"
```

## Design Goals

- Keep module boundaries clear
- Make each stage replaceable
- Keep the pipeline easy to understand
- Support gradual extension from a minimal working RAG system

## Current Status

Current implemented path:

- PDF loading is implemented
- CSV loading is implemented
- Chunk-based preprocessing is implemented
- Proposition-based preprocessing is implemented
- OpenRouter embedding is implemented
- In-memory vector indexing is implemented
- Cosine-similarity retrieval is implemented
- BM25 sparse retrieval is implemented
- Hybrid retrieval with RRF fusion is implemented
- Zhipu-based generation is implemented
- OpenRouter-based generation is implemented
- OpenRouter model selection is configurable from YAML
- DeepEval-style evaluation is implemented

Not yet expanded:

- Advanced pre-retrieval logic
- Advanced post-retrieval logic
- Persistent vector databases
- Multi-file ingestion workflows
