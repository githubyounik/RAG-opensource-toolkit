"""Factory helpers for pre-retrieval components."""

from __future__ import annotations

from typing import Any

from rag_toolkit.llm import create_chat_llm_client
from rag_toolkit.pre_retrieval.base import PreRetriever
from rag_toolkit.pre_retrieval.hyde import HyDEPreRetriever
from rag_toolkit.pre_retrieval.query_rewrite import QueryRewritePreRetriever
from rag_toolkit.pre_retrieval.query_transformer import QueryTransformer
from rag_toolkit.pre_retrieval.step_back import StepBackPreRetriever


def create_pre_retriever_from_config(
    pre_retrieval_config: dict[str, Any] | None,
    *,
    openrouter_api_key: str | None = None,
    zhipu_api_key: str | None = None,
) -> PreRetriever | None:
    """Create a pre-retrieval component from config.

    Supported strategies:
    - ``rewrite``
    - ``step_back``
    - ``hyde``
    """

    if not pre_retrieval_config or not bool(pre_retrieval_config.get("enabled", False)):
        return None

    strategy = str(pre_retrieval_config["strategy"]).lower()
    llm_client = create_chat_llm_client(
        pre_retrieval_config,
        openrouter_api_key=openrouter_api_key,
        zhipu_api_key=zhipu_api_key,
    )
    transformer = QueryTransformer(
        llm_client,
        temperature=float(pre_retrieval_config.get("temperature", 0.0)),
        max_tokens=(
            int(pre_retrieval_config["max_tokens"])
            if pre_retrieval_config.get("max_tokens") is not None
            else 256
        ),
    )

    if strategy == "rewrite":
        return QueryRewritePreRetriever(transformer)

    if strategy == "step_back":
        return StepBackPreRetriever(transformer)

    if strategy == "hyde":
        return HyDEPreRetriever(
            transformer,
            target_char_length=(
                int(pre_retrieval_config["hyde_target_char_length"])
                if pre_retrieval_config.get("hyde_target_char_length") is not None
                else None
            ),
        )

    raise ValueError(f"Unsupported pre-retrieval strategy: {strategy}")
