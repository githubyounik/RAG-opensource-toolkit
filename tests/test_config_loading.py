from rag_toolkit.core import load_pipeline_config


def test_pipeline_config_contains_chunk_settings() -> None:
    config = load_pipeline_config("configs/pipeline.example.yaml")

    document_processing_config = config["indexing"]["document_processing"]

    assert document_processing_config["chunk_size"] == 1000
    assert document_processing_config["chunk_overlap"] == 200
