"""
OpenAI-compatible LLM Client for Property Insurance RAG.

Handles API initialization, request dispatch, structured response parsing,
detailed request/response/usage logging, and robust error handling (401, 429, Connection errors).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import (
    APIConnectionError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from src.config import config

# Configure logger
logger = logging.getLogger("property_insurance_rag.llm_client")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class LLMError(Exception):
    """Base exception for LLM operations."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when authentication fails (HTTP 401)."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limit or quota is exceeded (HTTP 429)."""
    pass


class LLMConnectionError(LLMError):
    """Raised when network or server connection fails."""
    pass


class LLMResponseError(LLMError):
    """Raised on unexpected API errors."""
    pass


@dataclass
class TokenUsage:
    """Token usage statistics for an API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_api_usage(cls, usage: Any) -> "TokenUsage":
        if not usage:
            return cls()
        return cls(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )


@dataclass
class ChatResult:
    """Structured response from the chat completion endpoint."""
    content: str
    model: str
    usage: TokenUsage
    finish_reason: Optional[str] = None


def get_llm_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OpenAI:
    """
    Constructs and returns an OpenAI-compatible client.
    
    Args:
        base_url: Optional override for the base API URL (supports local/alternate providers).
        api_key: Optional override for the API key.
        
    Returns:
        OpenAI: Initialized client instance.
    """
    effective_base_url = base_url or config.OPENAI_BASE_URL
    effective_api_key = api_key or config.OPENAI_API_KEY

    return OpenAI(
        base_url=effective_base_url,
        api_key=effective_api_key or "placeholder_key",
    )


def send_chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    client: Optional[OpenAI] = None,
) -> ChatResult:
    """
    Sends a chat completion request to an OpenAI-compatible API,
    logs the request, response, and token usage, and handles standard HTTP errors.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.
        model: Target model name (defaults to config.CHAT_MODEL).
        temperature: Sampling temperature (0.0 for deterministic insurance reasoning).
        client: Optional OpenAI client instance.

    Returns:
        ChatResult: Extracted message content and usage metadata.

    Raises:
        LLMAuthenticationError: When API key is invalid or unauthorized (HTTP 401).
        LLMRateLimitError: When rate limits or quotas are exceeded (HTTP 429).
        LLMConnectionError: When API connection fails.
        LLMResponseError: For other unexpected API issues.
    """
    active_client = client or get_llm_client()
    target_model = model or config.CHAT_MODEL

    logger.info("REQUEST: Model=%s, Temperature=%s, Messages=%s", target_model, temperature, messages)

    try:
        response = active_client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
        )
    except AuthenticationError as exc:
        msg = "Auth failed (401): check OPENAI_API_KEY in your .env"
        logger.error("%s: %s", msg, exc)
        raise LLMAuthenticationError(msg) from exc
    except RateLimitError as exc:
        msg = "Rate limited (429): slow down and retry with backoff"
        logger.warning("%s: %s", msg, exc)
        raise LLMRateLimitError(msg) from exc
    except APIConnectionError as exc:
        msg = "Connection failed: unable to reach OpenAI-compatible endpoint"
        logger.error("%s: %s", msg, exc)
        raise LLMConnectionError(msg) from exc
    except OpenAIError as exc:
        msg = f"API error occurred: {exc}"
        logger.error(msg)
        raise LLMResponseError(msg) from exc
    except Exception as exc:
        msg = f"Unexpected error during chat completion: {exc}"
        logger.error(msg)
        raise LLMResponseError(msg) from exc

    if not response.choices:
        msg = "Empty response received: choices array is empty."
        logger.error(msg)
        raise LLMResponseError(msg)

    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = getattr(choice, "finish_reason", None)
    usage = TokenUsage.from_api_usage(response.usage)

    logger.info("RESPONSE: %s", content)
    logger.info("USAGE: %s", usage)

    return ChatResult(
        content=content,
        model=target_model,
        usage=usage,
        finish_reason=finish_reason,
    )


def ask_insurance_assistant(
    question: str,
    context: Optional[str] = None,
    system_prompt_path: Optional[Path] = None,
    client: Optional[OpenAI] = None,
) -> ChatResult:
    """
    Helper function to query the assistant with property insurance context.

    Args:
        question: Adjuster question regarding coverage, terms, or policy guidelines.
        context: Optional retrieved policy excerpts.
        system_prompt_path: Path to custom prompt file (defaults to prompts/system_prompt.txt).
        client: Optional OpenAI client instance.

    Returns:
        ChatResult: Grounded response with citations or strict refusal.
    """
    prompt_file = system_prompt_path or (config.PROMPTS_DIR / "system_prompt.txt")
    if prompt_file.exists():
        system_text = prompt_file.read_text(encoding="utf-8")
    else:
        system_text = "You are a concise property insurance assistant. Answer only with verifiable facts."

    user_content = question
    if context:
        user_content = f"Policy Context:\n{context}\n\nAdjuster Question:\n{question}"

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]

    return send_chat_completion(messages=messages, client=client)
