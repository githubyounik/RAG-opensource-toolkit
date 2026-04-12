# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -e ".[dev]"
```

## Commands

```bash
# Run all tests
pytest

# Run a single test
pytest tests/test_simple_rag_flow.py::test_simple_rag_pipeline_returns_context_and_metrics

# Lint
ruff check src tests

# Run the example (requires a PDF file)
PYTHONPATH=src python examples/simple_rag.py <pdf_path> "<query>"
```

## Architecture

The toolkit is a modular RAG pipeline where every stage is a separate module under `src/rag_toolkit/`. All modules share the same dataclass types from `core/types.py` for cross-module communication.

### Data flow through `RAGPipeline.run()`

```
Query → PreRetriever.process() → Retriever.retrieve() → PostRetriever.process() → Generator.generate() → Evaluator.evaluate()
         (optional)                (required)             (optional)                (required)             (optional)
```

`pipelines/rag_pipeline.py` wires these together. `PreRetriever`, `PostRetriever`, and `Evaluator` are optional at construction time.

### Core types (`core/types.py`)

All inter-module data is passed as these dataclasses (all use `slots=True`):
- `Query` → input to the pipeline
- `Document` → a single retrieved chunk
- `RetrievalResult` → output of retrieval/post-retrieval stages
- `GenerationResult` → output of generation
- `EvaluationResult` → output of evaluation

### Adding a new component

Each module has a `base.py` defining an abstract base class (e.g., `Retriever`, `Generator`). To implement a new component:
1. Subclass the base in the same module directory (e.g., `retrieval/my_retriever.py`)
2. Implement the required abstract method
3. Export it from the module's `__init__.py`

All base classes inherit from `core/base.py:Component`, which accepts an optional `ComponentConfig`.

### Indexing module

`indexing/` is slightly different from other modules — it has its own `InMemoryIndex` (a simple document store) and `SimplePDFIndexer` that reads PDFs via `pypdf` and chunks them into `Document` objects stored in the index. The index is passed directly to `SimpleRetriever` at construction time.

### Current implementations

All `Simple*` classes are minimal stubs. `SimpleRetriever` does keyword overlap scoring; `SimpleGenerator` concatenates retrieved document texts; `SimpleEvaluator` counts context documents. These exist for testing and as implementation examples, not for production use.
