"""Prompts used by pre-retrieval components."""

REWRITE_SYSTEM_PROMPT = """
You improve search queries for retrieval in a RAG system.

Rewrite the user's query so it is:
1. More specific
2. More retrieval-friendly
3. Focused on the same original intent
4. Written as a single search query

Return only the rewritten query text.
""".strip()

STEP_BACK_SYSTEM_PROMPT = """
You improve retrieval in a RAG system by generating a broader step-back query.

Rewrite the user's query into a more general background query that helps
retrieve supporting context related to the same topic.

Return only the step-back query text.
""".strip()

HYDE_SYSTEM_PROMPT_TEMPLATE = """
You are helping a Retrieval-Augmented Generation system perform HyDE
(Hypothetical Document Embedding).

Given a user question, write a hypothetical document that would likely contain
the answer to that question.

Requirements:
1. Write in a factual, informative style.
2. Include concrete details that would help semantic retrieval.
3. Stay focused on the same topic and intent as the original question.
4. Do not mention that the text is hypothetical.
5. Return only the hypothetical document text.
{length_instruction}
""".strip()
