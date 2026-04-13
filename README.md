# RAG Open Source Toolkit

A modular Python toolkit for building Retrieval-Augmented Generation (RAG) pipelines.

This repository is no longer only a framework skeleton. It now contains a minimal end-to-end RAG flow built around:

- PDF loading
- CSV loading
- Text cleaning and chunking
- Embedding-based indexing
- Embedding retrieval
- LLM generation
- Pipeline orchestration

## Current Architecture

The current codebase supports both PDF and CSV input, and both follow the same high-level pipeline:

```text
FileLoader
  -> DocumentProcessor
  -> OpenRouterEmbedder
  -> EmbeddingIndexer
  -> VectorIndex
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
  -> DocumentProcessor
  -> OpenRouterEmbedder
  -> EmbeddingIndexer
  -> VectorIndex
  -> EmbeddingRetriever
  -> configured Generator
  -> RAGPipeline
```

Example flow for a CSV file:

```text
CSVLoader
  -> DocumentProcessor
  -> OpenRouterEmbedder
  -> EmbeddingIndexer
  -> VectorIndex
  -> EmbeddingRetriever
  -> configured Generator
  -> RAGPipeline
```

In plain language, the pipeline does this:

1. Load a source file and extract structured text.
2. Clean and split the text into chunks.
3. Convert chunks into embeddings.
4. Store chunk embeddings in an in-memory vector index.
5. Embed the user query and retrieve the most similar chunks.
6. Send the retrieved context to a generation model.
7. Return the final answer through a unified pipeline interface.

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
- `DocumentProcessor`: cleans text and splits it into overlapping chunks
- `FileLoader`, `TextProcessor`: base interfaces for indexing-related components

### `embeddings`

Responsible for dense vector creation and vector index construction.

- `OpenRouterEmbedder`: calls embedding models through OpenRouter
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

### `pipelines`

Responsible for chaining modules together.

- `RAGPipeline`: standard orchestration entry for pre-retrieval, retrieval, post-retrieval, generation, and evaluation

### `pre_retrieval`, `post_retrieval`, `evaluation`

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
    chunk_size: 1000
    chunk_overlap: 200
```

You can change these values to control how the `DocumentProcessor` splits text before embedding and retrieval.

## Generation Model Configuration

The generation provider and model are also controlled through [configs/pipeline.example.yaml](/home/test/Desktop/code/RAG-opensource-toolkit/configs/pipeline.example.yaml).

Current default:

```yaml
generation:
  provider: openrouter
  model: nvidia/nemotron-3-super-120b-a12b:free
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
  model: google/gemma-4-26b-a4b-it:free
  temperature: 0.6
  max_tokens:
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
