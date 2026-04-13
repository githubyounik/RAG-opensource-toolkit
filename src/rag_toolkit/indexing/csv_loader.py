"""CSV file loader.

This loader converts a CSV file into the shared :class:`ParsedFile` format so
the rest of the RAG pipeline can stay unchanged. That is the key design goal:
add CSV support by reusing the existing indexing, embedding, retrieval, and
generation pipeline instead of creating a second parallel workflow.
"""

from __future__ import annotations

import csv
from pathlib import Path

from rag_toolkit.core.types import ParsedFile
from rag_toolkit.indexing.base import FileLoader


class CSVLoader(FileLoader):
    """Load a CSV file and represent each row as one text block.

    Why each row becomes one block:
    - A CSV file is naturally record-oriented.
    - One row usually describes one entity, such as one customer.
    - Keeping one row together makes retrieval more meaningful than splitting
      raw comma-separated text too early.

    The returned value is still a ``ParsedFile`` so downstream code can keep
    using the same `DocumentProcessor` that is already used for PDF input.
    """

    def load(self, path: str) -> ParsedFile:
        """Read a CSV file and convert rows into human-readable text strings.

        Each row is formatted as:
        ``column_a: value_a, column_b: value_b, ...``

        This gives embedding and retrieval stages more semantic text than raw
        CSV syntax, while still preserving the original row information.
        """

        csv_path = Path(path)
        rows_as_text: list[str] = []

        # DictReader lets us keep the CSV header names and map each row into a
        # dictionary. This makes it easy to render each record as readable text.
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames or []

            for row_index, row in enumerate(reader):
                # We explicitly keep the original row number in the text. This
                # helps the generated answer cite or reference the source record.
                parts = [f"row: {row_index}"]

                for column_name in fieldnames:
                    value = row.get(column_name, "")
                    # Collapse surrounding whitespace so the text sent to the
                    # embedder stays clean and stable.
                    cleaned_value = str(value).strip()
                    parts.append(f"{column_name}: {cleaned_value}")

                rows_as_text.append(", ".join(parts))

        return ParsedFile(
            source=csv_path.name,
            pages=rows_as_text,
            metadata={
                "path": str(csv_path),
                "row_count": len(rows_as_text),
                "columns": fieldnames,
            },
        )

