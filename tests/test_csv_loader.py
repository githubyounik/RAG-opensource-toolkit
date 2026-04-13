from pathlib import Path

from rag_toolkit.indexing import CSVLoader, DocumentProcessor


def test_csv_loader_returns_parsed_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "first_name,last_name,company\n"
        "Sheryl,Baxter,Acme Corp\n"
        "John,Smith,Blue Sky Ltd\n",
        encoding="utf-8",
    )

    loader = CSVLoader()
    parsed = loader.load(str(csv_path))

    assert parsed.source == "customers.csv"
    assert parsed.metadata["row_count"] == 2
    assert parsed.metadata["columns"] == ["first_name", "last_name", "company"]
    assert "first_name: Sheryl" in parsed.pages[0]
    assert "company: Acme Corp" in parsed.pages[0]


def test_csv_loader_output_can_be_processed_into_documents(tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "first_name,last_name,company\n"
        "Sheryl,Baxter,Acme Corp\n",
        encoding="utf-8",
    )

    loader = CSVLoader()
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)

    parsed = loader.load(str(csv_path))
    documents = processor.process(parsed)

    assert len(documents) == 1
    assert documents[0].metadata["source"] == "customers.csv"
    assert "Sheryl" in documents[0].text
