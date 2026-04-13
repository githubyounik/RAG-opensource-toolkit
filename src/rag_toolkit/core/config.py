"""Shared configuration helpers.

The current project only needs a very small amount of configuration logic.
This module keeps that logic in one place so example scripts do not each
re-implement YAML loading and nested key access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_pipeline_config(path: str = "configs/pipeline.example.yaml") -> dict[str, Any]:
    """Load a YAML config file and return it as a dictionary.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        The parsed configuration contents.

    Notes
    -----
    The examples use this helper to read chunking parameters such as
    ``chunk_size`` and ``chunk_overlap`` from the config file.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    if not isinstance(data, dict):
        raise ValueError("The pipeline configuration must be a YAML mapping.")

    return data
