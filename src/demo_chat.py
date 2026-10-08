"""
Interactive Demo & Verification Script for LLM Chat Completion.

Demonstrates:
1. Client initialization from environment config (.env).
2. Sending a chat completion request and reading choices[0].message.content.
3. Logging request payloads, response text, and token usage.
4. Catching and explaining common HTTP errors (401 Unauthorized, 429 Rate Limited).
"""

import argparse
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Allow running directly as python src/demo_chat.py or as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AuthenticationError, RateLimitError

from src.config import config
from src.llm_client import (
    ChatResult,
    LLMAuthenticationError,
    LLMRateLimitError,
    ask_insurance_assistant,
    get_llm_client,
    send_chat_completion,
)

# Configure console logger for demo
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%H:%M:%S",
)


def run_live_query(question: str) -> None:
    """Runs a live chat completion against configured endpoint."""
    print("=" * 70)
    print("RUNNING LIVE CHAT COMPLETION QUERY")
    print(f"Endpoint: {config.OPENAI_BASE_URL}")
    print(f"Model:    {config.CHAT_MODEL}")
    print(f"Query:    {question}")
    print("=" * 70)

    if not config.validate_api_keys():
        print("\n[!] WARNING: OPENAI_API_KEY is not set or using placeholder in .env.")
        print("    Configure your key in .env or run with --demo-all to see simulated tests.\n")

    try:
        result: ChatResult = ask_insurance_assistant(question)
        print("\n--- Model Output ---")
        print(result.content)
        print("\n--- Token Usage ---")
        print(f"Prompt Tokens:     {result.usage.prompt_tokens}")
        print(f"Completion Tokens: {result.usage.completion_tokens}")
        print(f"Total Tokens:      {result.usage.total_tokens}")
    except LLMAuthenticationError as err:
        print(f"\n[X] Caught Authentication Error: {err}")
    except LLMRateLimitError as err:
        print(f"\n[X] Caught Rate Limit Error: {err}")
    except Exception as err:
        print(f"\n[X] Error: {err}")


def simulate_error_scenarios() -> None:
    """Simulates 401 Unauthorized and 429 RateLimitError to demonstrate error handling."""
    print("\n" + "=" * 70)
    print("DEMONSTRATING ERROR HANDLING & LOGGING")
    print("=" * 70)

    # 1. Simulate 401 Unauthorized
    print("\n[Scenario 1: HTTP 401 - Invalid / Missing API Key]")
    mock_client_401 = MagicMock()
    mock_response = MagicMock(status_code=401, headers={})
    mock_client_401.chat.completions.create.side_effect = AuthenticationError(
        message="Incorrect API key provided.",
        response=mock_response,
        body={"error": {"message": "Invalid API key", "type": "invalid_request_error"}},
    )

    try:
        send_chat_completion(
            messages=[{"role": "user", "content": "Test 401"}],
            client=mock_client_401,
        )
    except LLMAuthenticationError as err:
        print(f"-> Safely caught and handled: {err}")

    # 2. Simulate 429 Rate Limit
    print("\n[Scenario 2: HTTP 429 - Rate Limit / Quota Exceeded]")
    mock_client_429 = MagicMock()
    mock_response_429 = MagicMock(status_code=429, headers={})
    mock_client_429.chat.completions.create.side_effect = RateLimitError(
        message="You exceeded your current quota.",
        response=mock_response_429,
        body={"error": {"message": "Quota exceeded", "type": "insufficient_quota"}},
    )

    try:
        send_chat_completion(
            messages=[{"role": "user", "content": "Test 429"}],
            client=mock_client_429,
        )
    except LLMRateLimitError as err:
        print(f"-> Safely caught and handled: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test and demonstrate LLM chat completion with logging & errors.")
    parser.add_argument(
        "--question",
        type=str,
        default="Does policy HO-3 cover sudden and accidental water damage from a ruptured pipe?",
        help="Adjuster question to ask the model.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulated 401 and 429 error scenarios.",
    )
    args = parser.parse_args()

    if args.simulate:
        simulate_error_scenarios()
    else:
        run_live_query(args.question)
        simulate_error_scenarios()


if __name__ == "__main__":
    main()
