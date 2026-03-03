"""Tests for statistical analysis module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.statistical import StatisticalAnalyzer


@pytest.fixture
def analyzer():
    return StatisticalAnalyzer()


def _make_logprobs(tokens_and_probs: list[tuple[str, float]]) -> list[dict]:
    """Helper: create logprob data from token-probability pairs."""
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


class TestKLDivergence:
    def test_identical_distributions(self, analyzer):
        lps = _make_logprobs([("a", -0.5), ("b", -1.0), ("c", -0.3)])
        kl = analyzer.kl_divergence(lps, lps)
        assert kl < 0.01, f"KL of identical distributions should be ~0, got {kl}"

    def test_different_distributions(self, analyzer):
        lps_a = _make_logprobs([("a", -0.1), ("b", -0.1), ("c", -0.1)])
        lps_b = _make_logprobs([("x", -0.1), ("y", -0.1), ("z", -0.1)])
        kl = analyzer.kl_divergence(lps_a, lps_b)
        assert kl > 0.1, f"KL of different distributions should be >0.1, got {kl}"

    def test_empty_logprobs(self, analyzer):
        kl = analyzer.kl_divergence([], [])
        assert kl == float("inf")

    def test_different_lengths(self, analyzer):
        lps_a = _make_logprobs([("a", -0.5), ("b", -1.0)])
        lps_b = _make_logprobs([("a", -0.5)])
        kl = analyzer.kl_divergence(lps_a, lps_b)
        assert np.isfinite(kl)


class TestShannonEntropy:
    def test_low_entropy(self, analyzer):
        # One dominant token
        lps = [{
            "top_logprobs": [
                {"token": "a", "logprob": -0.01},  # prob ≈ 0.99
                {"token": "b", "logprob": -5.0},    # prob ≈ 0.007
            ]
        }]
        entropies = analyzer.shannon_entropy(lps)
        assert len(entropies) == 1
        assert entropies[0] < 0.5

    def test_high_entropy(self, analyzer):
        # Uniform-ish distribution
        lps = [{
            "top_logprobs": [
                {"token": "a", "logprob": -1.6},
                {"token": "b", "logprob": -1.6},
                {"token": "c", "logprob": -1.6},
                {"token": "d", "logprob": -1.6},
                {"token": "e", "logprob": -1.6},
            ]
        }]
        entropies = analyzer.shannon_entropy(lps)
        assert entropies[0] > 1.0

    def test_empty(self, analyzer):
        assert analyzer.shannon_entropy([]) == []


class TestTokenAgreement:
    def test_perfect_agreement(self, analyzer):
        lps = _make_logprobs([("a", -0.5), ("b", -1.0)])
        rate = analyzer.token_agreement_rate(lps, lps)
        assert rate == 1.0

    def test_no_agreement(self, analyzer):
        lps_a = _make_logprobs([("a", -0.5), ("b", -1.0)])
        lps_b = _make_logprobs([("x", -0.5), ("y", -1.0)])
        rate = analyzer.token_agreement_rate(lps_a, lps_b)
        assert rate == 0.0

    def test_partial_agreement(self, analyzer):
        lps_a = _make_logprobs([("a", -0.5), ("b", -1.0)])
        lps_b = _make_logprobs([("a", -0.5), ("y", -1.0)])
        rate = analyzer.token_agreement_rate(lps_a, lps_b)
        assert rate == 0.5


class TestBootstrapCI:
    def test_tight_ci(self, analyzer):
        values = [1.0] * 100
        mean, lower, upper = analyzer.bootstrap_confidence_interval(values)
        assert abs(mean - 1.0) < 0.01
        assert abs(upper - lower) < 0.01

    def test_wide_ci(self, analyzer):
        np.random.seed(42)
        values = list(np.random.normal(0, 10, 20))
        mean, lower, upper = analyzer.bootstrap_confidence_interval(values)
        assert lower < mean < upper


class TestLogprobStats:
    def test_top1_stats(self, analyzer):
        lps = _make_logprobs([("a", -0.5), ("b", -1.5), ("c", -1.0)])
        stats = analyzer.top1_logprob_stats(lps)
        assert stats["n"] == 3
        assert abs(stats["mean"] - (-1.0)) < 0.01

    def test_gap_stats(self, analyzer):
        lps = _make_logprobs([("a", -0.5), ("b", -1.0)])
        stats = analyzer.logprob_gap_stats(lps)
        assert stats["n"] == 2
        assert stats["mean"] == 2.0  # each gap is 2.0 (from _make_logprobs)
