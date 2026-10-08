"""
Unit tests for LLM Chat Completion client, logging, and error handling.
Tests run offline without requiring paid API credentials.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from openai import (
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
)

from src.llm_client import (
    ChatResult,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    ask_insurance_assistant,
    get_llm_client,
    send_chat_completion,
)


def create_mock_completion_response(content: str, prompt_tokens: int = 15, completion_tokens: int = 25):
    """Helper to mock OpenAI API chat completion response."""
    choice_mock = MagicMock()
    choice_mock.message.content = content
    choice_mock.finish_reason = "stop"

    usage_mock = MagicMock()
    usage_mock.prompt_tokens = prompt_tokens
    usage_mock.completion_tokens = completion_tokens
    usage_mock.total_tokens = prompt_tokens + completion_tokens

    response_mock = MagicMock()
    response_mock.choices = [choice_mock]
    response_mock.usage = usage_mock
    return response_mock


def test_get_llm_client_configuration():
    """Verify client instantiates with specified or configured endpoints."""
    custom_url = "http://localhost:11434/v1"
    custom_key = "test-sk-key"
    client = get_llm_client(base_url=custom_url, api_key=custom_key)
    assert str(client.base_url).rstrip("/") == custom_url.rstrip("/")
    assert client.api_key == custom_key


def test_send_chat_completion_success(caplog):
    """Test successful chat completion, response parsing, and logging."""
    mock_client = MagicMock()
    mock_response = create_mock_completion_response(
        content="Water damage is covered subject to policy deductible.",
        prompt_tokens=42,
        completion_tokens=12,
    )
    mock_client.chat.completions.create.return_value = mock_response

    messages = [
        {"role": "system", "content": "You are an adjuster assistant."},
        {"role": "user", "content": "Is burst pipe covered under HO-3?"},
    ]

    with caplog.at_level(logging.INFO):
        result: ChatResult = send_chat_completion(
            messages=messages,
            model="gpt-4o-mini",
            client=mock_client,
        )

    # Verify returned result structure
    assert result.content == "Water damage is covered subject to policy deductible."
    assert result.usage.prompt_tokens == 42
    assert result.usage.completion_tokens == 12
    assert result.usage.total_tokens == 54
    assert result.finish_reason == "stop"

    # Verify logging of REQUEST, RESPONSE, and USAGE
    logs = caplog.text
    assert "REQUEST: Model=gpt-4o-mini" in logs
    assert "RESPONSE: Water damage is covered subject to policy deductible." in logs
    assert "USAGE: TokenUsage(prompt_tokens=42, completion_tokens=12, total_tokens=54)" in logs


def test_send_chat_completion_authentication_error_401():
    """Test handling and descriptive error message for HTTP 401 Unauthorized."""
    mock_client = MagicMock()
    mock_response = MagicMock(status_code=401, headers={})
    mock_client.chat.completions.create.side_effect = AuthenticationError(
        message="Incorrect API key provided.",
        response=mock_response,
        body={"error": {"message": "Invalid API key"}},
    )

    with pytest.raises(LLMAuthenticationError) as exc_info:
        send_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            client=mock_client,
        )

    assert "Auth failed (401)" in str(exc_info.value)
    assert "check OPENAI_API_KEY in your .env" in str(exc_info.value)


def test_send_chat_completion_rate_limit_error_429():
    """Test handling and descriptive error message for HTTP 429 Rate Limit."""
    mock_client = MagicMock()
    mock_response = MagicMock(status_code=429, headers={})
    mock_client.chat.completions.create.side_effect = RateLimitError(
        message="Rate limit reached.",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )

    with pytest.raises(LLMRateLimitError) as exc_info:
        send_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            client=mock_client,
        )

    assert "Rate limited (429)" in str(exc_info.value)
    assert "slow down and retry with backoff" in str(exc_info.value)


def test_send_chat_completion_connection_error():
    """Test handling for API connection / network errors."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(
        request=MagicMock()
    )

    with pytest.raises(LLMConnectionError) as exc_info:
        send_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            client=mock_client,
        )

    assert "Connection failed" in str(exc_info.value)


def test_send_chat_completion_empty_choices():
    """Test defensive handling when API returns empty choices."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = []
    mock_client.chat.completions.create.return_value = mock_response

    with pytest.raises(LLMResponseError) as exc_info:
        send_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            client=mock_client,
        )

    assert "Empty response received" in str(exc_info.value)


def test_ask_insurance_assistant():
    """Test helper for insurance assistant query formatting."""
    mock_client = MagicMock()
    mock_response = create_mock_completion_response("Coverage confirmed under rider 14B.")
    mock_client.chat.completions.create.return_value = mock_response

    result = ask_insurance_assistant(
        question="Is roof replacement covered?",
        context="Rider 14B includes replacement cost for windstorm.",
        client=mock_client,
    )

    assert result.content == "Coverage confirmed under rider 14B."
    called_messages = mock_client.chat.completions.create.call_args[1]["messages"]
    assert len(called_messages) == 2
    assert called_messages[0]["role"] == "system"
    assert "Policy Context:\nRider 14B" in called_messages[1]["content"]
    assert "Adjuster Question:\nIs roof replacement covered?" in called_messages[1]["content"]
