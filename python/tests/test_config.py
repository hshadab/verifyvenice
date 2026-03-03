"""Tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import VerifyConfig


@pytest.fixture
def sample_config():
    return {
        "venice": {
            "base_url": "https://api.venice.ai/api/v1",
            "model_70b": "llama-3.3-70b",
            "model_3b": "llama-3.2-3b",
        },
        "reference": {
            "provider": "together",
            "base_url": "https://api.together.xyz/v1",
            "model_70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        },
        "local": {
            "base_url": "http://localhost:11434",
            "model_3b": "llama3.2:3b",
        },
        "test_params": {
            "repetitions": 3,
            "top_logprobs": 5,
            "max_tokens": 64,
            "temperature": 0.0,
            "seeds": [42, 137, 256],
            "rate_limit_sleep": 0.1,
            "output_integrity_prompts": {"factual": "What is 2+2?"},
            "model_authenticity_prompts": {"probe": "Tell me about AI."},
        },
        "thresholds": {
            "kl_divergence_warn": 0.5,
            "kl_divergence_fail": 2.0,
            "token_agreement_min": 0.60,
            "entropy_ratio_range": [0.5, 2.0],
            "cosine_similarity_min": 0.85,
        },
        "calibration": {
            "local_repetitions": 5,
            "expected_kl_ranges": {
                "same_model_same_quant": [0.0, 0.05],
                "same_model_diff_quant": [0.05, 0.30],
                "different_model": [1.0, 100.0],
            },
        },
    }


@pytest.fixture
def config_files(sample_config, tmp_path):
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"

    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)

    with open(env_path, "w") as f:
        f.write("VENICE_API_KEY=test-venice-key\n")
        f.write("TOGETHER_API_KEY=test-together-key\n")

    return config_path, env_path


def test_load_config(config_files):
    config_path, env_path = config_files
    config = VerifyConfig.load(config_path, env_path)

    assert config.venice.base_url == "https://api.venice.ai/api/v1"
    assert config.venice.model_70b == "llama-3.3-70b"
    assert config.reference.provider == "together"
    assert config.test_params.repetitions == 3
    assert config.test_params.temperature == 0.0
    assert len(config.test_params.seeds) == 3


def test_validate_config(config_files):
    config_path, env_path = config_files
    config = VerifyConfig.load(config_path, env_path)
    errors = config.validate()
    assert len(errors) == 0


def test_validate_missing_keys(config_files, monkeypatch):
    config_path, _ = config_files
    # Create .env without keys
    empty_env = config_path.parent / ".env_empty"
    empty_env.touch()

    # Clear any real keys from environment
    monkeypatch.delenv("VENICE_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    config = VerifyConfig.load(config_path, empty_env)
    errors = config.validate()
    assert any("VENICE_API_KEY" in e for e in errors)
    assert any("TOGETHER_API_KEY" in e for e in errors)


def test_get_seeds(config_files):
    config_path, env_path = config_files
    config = VerifyConfig.load(config_path, env_path)

    seeds = config.get_seeds(2)
    assert len(seeds) == 2
    assert seeds == [42, 137]

    all_seeds = config.get_seeds()
    assert len(all_seeds) == 3
