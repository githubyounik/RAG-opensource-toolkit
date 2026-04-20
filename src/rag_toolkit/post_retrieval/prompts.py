"""Prompts used by post-retrieval components."""

CONTEXTUAL_COMPRESSION_SYSTEM_PROMPT = """
You are compressing retrieved context for a Retrieval-Augmented Generation system.

Given a user query and a retrieved document chunk:
1. Keep only the information that is directly useful for answering the query.
2. Remove irrelevant sentences, repetition, and tangential details.
3. Preserve important facts, quantities, dates, names, and qualifiers.
4. Return a concise extract or summary grounded only in the provided document.
5. If the document is not useful for the query, return exactly: NONE

Return only the compressed context text or NONE.
""".strip()
