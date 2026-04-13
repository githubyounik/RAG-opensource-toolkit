# RAG Open Source Toolkit

A modular Python toolkit for building Retrieval-Augmented Generation (RAG) pipelines.

This repository is no longer only a framework skeleton. It now contains a minimal end-to-end RAG flow built around:

- PDF loading
- CSV loading
- Text cleaning and chunking
- Pre-retrieval query transformation
- Embedding-based indexing
- Embedding retrieval
- LLM generation
- Pipeline orchestration

## Current Architecture

The current codebase supports both PDF and CSV input, and both follow the same high-level pipeline:

```text
FileLoader
  -> configured TextProcessor
  -> configured Embedder
  -> EmbeddingIndexer
  -> VectorIndex
  -> optional configured PreRetriever
  -> EmbeddingRetriever
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
  -> configured Embedder
  -> EmbeddingIndexer
  -> VectorIndex
  -> optional configured PreRetriever
  -> EmbeddingRetriever
  -> configured Generator
  -> RAGPipeline
```

Example flow for a CSV file:

```text
CSVLoader
  -> configured TextProcessor
  -> configured Embedder
  -> EmbeddingIndexer
  -> VectorIndex
  -> optional configured PreRetriever
  -> EmbeddingRetriever
  -> configured Generator
  -> RAGPipeline
```

In plain language, the pipeline does this:

1. Load a source file and extract structured text.
2. Clean and split the text into chunks.
3. Convert chunks into embeddings.
4. Store chunk embeddings in an in-memory vector index.
5. Optionally rewrite or broaden the user query before retrieval.
6. Embed the query and retrieve the most similar chunks.
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

These modules currently keep their base interfaces and are reserved for later extension.

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
- Zhipu-based generation is implemented
- OpenRouter-based generation is implemented
- OpenRouter model selection is configurable from YAML

Not yet expanded:

- Advanced pre-retrieval logic
- Advanced post-retrieval logic
- Rich evaluation implementations
- Persistent vector databases
- Multi-file ingestion workflows
