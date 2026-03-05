"""Shared fixtures and helpers for VerifyVenice tests."""

from unittest.mock import MagicMock

import pytest

from src.analysis.fingerprint import ModelFingerprinter
from src.analysis.similarity import TextSimilarity
from src.analysis.statistical import StatisticalAnalyzer


@pytest.fixture
def analyzer():
    return StatisticalAnalyzer()


@pytest.fixture
def similarity():
    return TextSimilarity()


@pytest.fixture
def fingerprinter():
    return ModelFingerprinter()


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.venice.base_url = "https://api.venice.ai/api/v1"
    cfg.venice.model_70b = "llama-3.3-70b"
    cfg.venice.model_3b = "llama-3.2-3b"
    cfg.venice.logprob_models = ["llama-3.2-3b"]
    cfg.reference.provider = "together"
    cfg.reference.base_url = "https://api.together.xyz/v1"
    cfg.reference.model_70b = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    cfg.local.base_url = "http://localhost:11434"
    cfg.local.model_3b = "llama3.2:3b"
    cfg.test_params.repetitions = 2
    cfg.test_params.top_logprobs = 5
    cfg.test_params.max_tokens = 512
    cfg.test_params.temperature = 0.0
    cfg.test_params.seeds = [42, 123, 456]
    cfg.test_params.rate_limit_sleep = 0.0
    cfg.test_params.prompts = {
        "factual": "What is photosynthesis?",
        "reasoning": "Explain why the sky is blue.",
    }
    cfg.test_params.output_integrity_prompts = cfg.test_params.prompts
    cfg.test_params.model_authenticity_prompts = cfg.test_params.prompts
    cfg.thresholds.kl_divergence_warn = 0.5
    cfg.thresholds.kl_divergence_fail = 2.0
    cfg.thresholds.token_agreement_min = 0.60
    cfg.thresholds.entropy_ratio_range = [0.5, 2.0]
    cfg.thresholds.cosine_similarity_min = 0.85
    cfg.calibration.local_repetitions = 3
    cfg.calibration.expected_kl_ranges = {"q4": [0.0, 0.5], "q8": [0.0, 0.2]}
    cfg.get_seeds.return_value = [42, 123, 456]
    cfg.venice_api_key = "test-key"
    cfg.together_api_key = "test-key"
    return cfg


def make_logprobs(tokens_and_probs):
    """Create logprob dicts from (token, logprob) pairs.

    Each entry gets a top_logprobs list with the token and one alternative.
    """
    result = []
    for token, logprob in tokens_and_probs:
        result.append({
            "token": token,
            "logprob": logprob,
            "top_logprobs": [
                {"token": token, "logprob": logprob},
                {"token": f"alt_{token}", "logprob": logprob - 2.0},
            ],
        })
    return result


def make_response_dict(
    response_text="This is a test response.",
    logprobs=None,
    finish_reason="stop",
    mean_lp=None,
    n_tokens=10,
):
    """Create a synthetic API response dict.

    If logprobs is None and mean_lp is provided, generates logprob data
    with the given mean logprob value.
    """
    if logprobs is None and mean_lp is not None:
        tokens = [f"tok{i}" for i in range(n_tokens)]
        logprobs = make_logprobs([(t, mean_lp + (i % 3) * 0.1) for i, t in enumerate(tokens)])
    return {
        "response_text": response_text,
        "logprobs": logprobs or [],
        "finish_reason": finish_reason,
        "usage": {"prompt_tokens": 20, "completion_tokens": n_tokens},
        "hashes": {},
        "latency": 0.5,
    }
