from rag_toolkit.pipelines import RAGPipeline


def test_pipeline_symbol_is_importable() -> None:
    assert RAGPipeline is not None
