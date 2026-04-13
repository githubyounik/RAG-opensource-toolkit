from rag_toolkit.llm.openrouter_client import OpenRouterChatClient


def test_openrouter_client_raises_clear_error_for_null_content() -> None:
    client = OpenRouterChatClient(api_key="test-key", model="z-ai/glm-5.1")

    response_json = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": None,
                },
            }
        ],
        "usage": {
            "completion_tokens_details": {
                "reasoning_tokens": 277,
            }
        },
    }

    try:
        client._extract_message_text(response_json)
    except RuntimeError as exc:
        message = str(exc)
        assert "contained no text content" in message
        assert "finish_reason=length" in message
        assert "reasoning_tokens=277" in message
        assert "increasing max_tokens" in message
    else:
        raise AssertionError("Expected RuntimeError for null content response.")
