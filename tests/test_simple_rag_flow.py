from rag_toolkit.core.types import Document, Query
from rag_toolkit.evaluation import SimpleEvaluator
from rag_toolkit.generation import SimpleGenerator
from rag_toolkit.indexing import InMemoryIndex
from rag_toolkit.pipelines import RAGPipeline
from rag_toolkit.post_retrieval import SimplePostRetriever
from rag_toolkit.pre_retrieval import SimplePreRetriever
from rag_toolkit.retrieval import SimpleRetriever


def test_simple_rag_pipeline_returns_context_and_metrics() -> None:
    index = InMemoryIndex(
        documents=[
            Document(doc_id="1", text="Climate change is mainly caused by greenhouse gas emissions."),
            Document(doc_id="2", text="Oceans absorb a large amount of the Earth's heat."),
        ]
    )

    pipeline = RAGPipeline(
        pre_retriever=SimplePreRetriever(),
        retriever=SimpleRetriever(index=index, top_k=2),
        post_retriever=SimplePostRetriever(),
        generator=SimpleGenerator(),
        evaluator=SimpleEvaluator(),
    )

    generation_result, evaluation_result = pipeline.run(
        Query(text="What causes climate change?")
    )

    assert "greenhouse gas emissions" in generation_result.answer.lower()
    assert evaluation_result is not None
    assert evaluation_result.metrics["context_count"] >= 1.0
