"""Save pipeline run results to a timestamped JSON file.

Each run produces one file under ``log_dir`` named after the UTC timestamp
at the moment of saving, e.g. ``2026-04-20T14-05-32.json``.

Saved fields
------------
- ``timestamp``      : ISO-8601 UTC timestamp
- ``command``        : the full command that was used to start the process
- ``config``         : the loaded pipeline YAML config (no secrets)
- ``query``          : the original question text
- ``answer``         : the generated answer
- ``retrieved_chunks``: list of retrieved documents with id, text, and metadata
- ``evaluation``     : metric scores and reasons (omitted when not available)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_toolkit.core.types import EvaluationResult, GenerationResult


def save_run_log(
    *,
    config: dict,
    query_text: str,
    generation_result: GenerationResult,
    evaluation_result: EvaluationResult | None = None,
    log_dir: str = "run_logs",
) -> str:
    """Persist one pipeline run to disk and return the file path."""

    timestamp = datetime.now(timezone.utc)
    # Use hyphens instead of colons so the name is valid on all platforms.
    filename = timestamp.strftime("%Y-%m-%dT%H-%M-%S") + ".json"

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    output_file = log_path / filename

    chunks = []
    for doc in (generation_result.contexts or []):
        chunks.append(
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata,
            }
        )

    record: dict = {
        "timestamp": timestamp.isoformat(),
        "command": " ".join(sys.argv),
        "config": config,
        "query": query_text,
        "answer": generation_result.answer,
        "retrieved_chunks": chunks,
    }

    if evaluation_result is not None:
        record["evaluation"] = {
            "metrics": evaluation_result.metrics,
            "metadata": evaluation_result.metadata,
        }

    output_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_file)
