from rag_toolkit.core.types import Document, Query, RetrievalResult
from rag_toolkit.post_retrieval.contextual_compressor import ContextualCompressor


class DummyClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
    ):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        class Response:
            def __init__(self, text: str) -> None:
                self.text = text

        return Response(self.outputs.pop(0))


def test_contextual_compressor_keeps_only_compressed_relevant_context() -> None:
    client = DummyClient(
        [
            "Sea levels have risen by about 20 centimeters in the past century.",
            "NONE",
        ]
    )
    compressor = ContextualCompressor(client, temperature=0.0, max_tokens=256)

    result = RetrievalResult(
        query=Query(text="How much have sea levels risen?"),
        documents=[
            Document(doc_id="doc-1", text="Long chunk about sea level rise.", metadata={}),
            Document(doc_id="doc-2", text="Irrelevant chunk about forests.", metadata={}),
        ],
    )

    compressed = compressor.process(result)

    assert len(compressed.documents) == 1
    assert compressed.documents[0].text == (
        "Sea levels have risen by about 20 centimeters in the past century."
    )
    assert compressed.documents[0].metadata["post_retrieval_strategy"] == (
        "contextual_compression"
    )
    assert compressed.metadata["compressed_document_count"] == 1
