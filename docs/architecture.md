# Architecture

This document describes the intended high-level architecture.

## Design Principles

- Separate each RAG stage into an independent module
- Keep module interfaces small and explicit
- Prefer composition over tightly coupled implementations
- Make testing and replacement easy

## Core Layers

- `core`: shared abstractions and data models
- `indexing`: corpus processing and index construction
- `pre_retrieval`: query rewriting, routing, planning, and filtering before retrieval
- `retrieval`: recall-oriented retrieval interfaces
- `post_retrieval`: reranking, compression, filtering, and context shaping
- `generation`: model-facing answer generation
- `evaluation`: offline or online quality assessment
- `pipelines`: orchestration across modules

