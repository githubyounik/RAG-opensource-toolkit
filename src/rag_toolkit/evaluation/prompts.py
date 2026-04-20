"""Prompts used by evaluation components."""

CORRECTNESS_SYSTEM_PROMPT = """
You are evaluating answer correctness.

Compare the generated answer against the reference answer. Focus on factual
agreement, not wording style. Return a score between 0 and 1.

Return exactly in this format:
SCORE: <number between 0 and 1>
REASON: <one short sentence>
""".strip()

FAITHFULNESS_SYSTEM_PROMPT = """
You are evaluating answer faithfulness for a RAG system.

Decide how well the generated answer is grounded in the retrieved context.
If the answer contains claims not supported by the context, reduce the score.
Return a score between 0 and 1.

Return exactly in this format:
SCORE: <number between 0 and 1>
REASON: <one short sentence>
""".strip()

CONTEXTUAL_RELEVANCY_SYSTEM_PROMPT = """
You are evaluating contextual relevancy for a RAG system.

Decide how relevant the retrieved context is for answering the user's question.
Return a score between 0 and 1.

Return exactly in this format:
SCORE: <number between 0 and 1>
REASON: <one short sentence>
""".strip()
