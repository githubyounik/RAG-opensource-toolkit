import json

from rag_toolkit.core.types import ParsedFile
from rag_toolkit.indexing import PropositionProcessor


def test_parse_propositions_returns_clean_strings() -> None:
    processor = PropositionProcessor(api_key="test-key")
    content = json.dumps(
        {
            "propositions": [
                "Paul Graham published Founder Mode in September 2024.",
                "Founder Mode challenges conventional startup management advice.",
            ]
        }
    )

    propositions = processor._parse_propositions(content)

    assert propositions == [
        "Paul Graham published Founder Mode in September 2024.",
        "Founder Mode challenges conventional startup management advice.",
    ]


def test_process_keeps_parent_metadata() -> None:
    class DummyPropositionProcessor(PropositionProcessor):
        def _request_completion(self, payload: dict[str, object]) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "propositions": [
                                        "Paul Graham published Founder Mode in September 2024."
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

    processor = DummyPropositionProcessor(
        api_key="test-key",
        base_chunk_size=200,
        base_chunk_overlap=20,
    )
    parsed_file = ParsedFile(
        source="essay.txt",
        pages=["Founder Mode was published in September 2024 by Paul Graham."],
        metadata={"path": "essay.txt"},
    )

    proposition_documents = processor.process(parsed_file)

    assert len(proposition_documents) == 1
    assert proposition_documents[0].doc_id == "essay.txt-chunk-0-prop-0"
    assert proposition_documents[0].metadata["parent_doc_id"] == "essay.txt-chunk-0"
    assert proposition_documents[0].metadata["chunking_strategy"] == "proposition"
