from rag_toolkit.core.types import Query
from rag_toolkit.pre_retrieval.hyde import HyDEPreRetriever
from rag_toolkit.pre_retrieval.query_rewrite import QueryRewritePreRetriever
from rag_toolkit.pre_retrieval.step_back import StepBackPreRetriever


class DummyTransformer:
    def __init__(self, output: str) -> None:
        self.output = output

    def transform(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.output


def test_query_rewrite_returns_new_query_text() -> None:
    pre_retriever = QueryRewritePreRetriever(
        DummyTransformer("What are the environmental impacts of climate change?")
    )

    query = Query(text="climate change impacts?")
    transformed = pre_retriever.process(query)

    assert transformed.text == "What are the environmental impacts of climate change?"
    assert transformed.metadata["original_query"] == "climate change impacts?"
    assert transformed.metadata["pre_retrieval_strategy"] == "rewrite"


def test_step_back_returns_new_query_text() -> None:
    pre_retriever = StepBackPreRetriever(
        DummyTransformer("What are the general effects of climate change?")
    )

    query = Query(text="How does climate change affect biodiversity?")
    transformed = pre_retriever.process(query)

    assert transformed.text == "What are the general effects of climate change?"
    assert transformed.metadata["original_query"] == "How does climate change affect biodiversity?"
    assert transformed.metadata["pre_retrieval_strategy"] == "step_back"


def test_hyde_returns_hypothetical_document_as_query_text() -> None:
    hypothetical_document = (
        "Sea level rise has increased over the past century due to thermal expansion "
        "and melting land ice, with measurements showing roughly 20 centimeters."
    )
    pre_retriever = HyDEPreRetriever(
        DummyTransformer(hypothetical_document),
        target_char_length=500,
    )

    query = Query(text="Sea levels have risen how much in the past century?")
    transformed = pre_retriever.process(query)

    assert transformed.text == hypothetical_document
    assert transformed.metadata["original_query"] == query.text
    assert transformed.metadata["pre_retrieval_strategy"] == "hyde"
    assert transformed.metadata["hyde_hypothetical_document"] == hypothetical_document
