"""Prompts used by indexing components."""

PROPOSITION_SYSTEM_PROMPT = """
Break the input text into simple, factual, self-contained propositions.

Rules:
1. Each proposition must express a single fact.
2. Each proposition must be understandable without additional context.
3. Use full names instead of pronouns when possible.
4. Preserve important dates, quantities, and qualifiers.
5. Keep each proposition concise.
6. Return valid JSON only.

Output format:
{
  "propositions": [
    "First proposition.",
    "Second proposition."
  ]
}
""".strip()
