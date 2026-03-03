"""Tests for API client response parsing.

These tests use mock responses to verify parsing logic
without making actual API calls.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _mock_config():
    """Create a minimal mock config for client initialization."""
    config = MagicMock()
    config.venice_api_key = "test-key"
    config.together_api_key = "test-key"
    config.venice.base_url = "https://api.venice.ai/api/v1"
    config.venice.model_70b = "llama-3.3-70b"
    config.venice.model_3b = "llama-3.2-3b"
    config.reference.base_url = "https://api.together.xyz/v1"
    config.reference.model_70b = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    config.reference.provider = "together"
    config.local.base_url = "http://localhost:11434"
    config.local.model_3b = "llama3.2:3b"
    config.test_params.top_logprobs = 5
    config.test_params.max_tokens = 64
    config.test_params.temperature = 0.0
    config.test_params.rate_limit_sleep = 0.0
    return config


def _mock_openai_response(text="Hello", model="llama-3.3-70b"):
    """Create a mock OpenAI chat completion response."""
    token_lp = MagicMock()
    token_lp.token = "Hello"
    token_lp.logprob = -0.5
    top_lp = MagicMock()
    top_lp.token = "Hello"
    top_lp.logprob = -0.5
    token_lp.top_logprobs = [top_lp]

    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = "stop"
    choice.logprobs.content = [token_lp]

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = usage

    return response


class TestVeniceClientParsing:
    @patch("src.clients.venice.OpenAI")
    def test_chat_completion_parsing(self, mock_openai_cls):
        from src.clients.venice import VeniceClient

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_openai_cls.return_value = mock_client

        config = _mock_config()
        client = VeniceClient(config)
        result = client.chat_completion("test prompt")

        assert result["provider"] == "venice"
        assert result["response_text"] == "Hello"
        assert len(result["logprobs"]) == 1
        assert result["logprobs"][0]["token"] == "Hello"
        assert result["logprobs"][0]["logprob"] == -0.5
        assert result["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 15
        assert "prompt_hash" in result
        assert "response_hash" in result
        assert result["latency_seconds"] >= 0


class TestReferenceClientParsing:
    @patch("src.clients.reference.OpenAI")
    def test_chat_completion_parsing(self, mock_openai_cls):
        from src.clients.reference import ReferenceClient

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo"
        )
        mock_openai_cls.return_value = mock_client

        config = _mock_config()
        client = ReferenceClient(config)
        result = client.chat_completion("test prompt")

        assert result["provider"] == "together"
        assert result["response_text"] == "Hello"
