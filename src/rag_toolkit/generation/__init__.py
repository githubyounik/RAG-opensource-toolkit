"""Generation module interfaces."""

from rag_toolkit.generation.base import Generator
from rag_toolkit.generation.zhipu_generator import ZhipuGenerator

__all__ = ["Generator", "ZhipuGenerator"]
