"""Base interfaces shared across the toolkit."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field


@dataclass(slots=True)
class ComponentConfig:
    """Base config object for any pluggable component."""

    name: str
    parameters: dict[str, object] = field(default_factory=dict)


class Component(ABC):
    """Base class for all toolkit components."""

    def __init__(self, config: ComponentConfig | None = None) -> None:
        self.config = config or ComponentConfig(name=self.__class__.__name__)
