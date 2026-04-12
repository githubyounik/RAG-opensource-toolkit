"""Quick test: verify OpenRouter chat completions work with google/gemma-4-26b-a4b-it:free."""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()


def test_openrouter_chat():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    assert api_key, "OPENROUTER_API_KEY not set in .env"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # First call
    resp = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(
            {
                "model": "google/gemma-4-26b-a4b-it:free",
                "messages": [
                    {"role": "user", "content": "How many r's are in the word 'strawberry'?"}
                ],
                "reasoning": {"enabled": True},
            }
        ),
        timeout=30,
    )

    assert resp.status_code == 200, f"First call failed: {resp.status_code} {resp.text}"
    msg = resp.json()["choices"][0]["message"]
    print("\n=== First response ===")
    print("Content:", msg.get("content"))
    print("Reasoning details:", msg.get("reasoning_details"))

    # Second call — continue from previous reasoning
    messages = [
        {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
        {
            "role": "assistant",
            "content": msg.get("content"),
            "reasoning_details": msg.get("reasoning_details"),
        },
        {"role": "user", "content": "Are you sure? Think carefully."},
    ]

    resp2 = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(
            {
                "model": "google/gemma-4-26b-a4b-it:free",
                "messages": messages,
                "reasoning": {"enabled": True},
            }
        ),
        timeout=30,
    )

    assert resp2.status_code == 200, f"Second call failed: {resp2.status_code} {resp2.text}"
    msg2 = resp2.json()["choices"][0]["message"]
    print("\n=== Second response ===")
    print("Content:", msg2.get("content"))
    print("Reasoning details:", msg2.get("reasoning_details"))
