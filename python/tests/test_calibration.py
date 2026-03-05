"""Tests for quantization calibration (src/analysis/calibration.py)."""

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from src.analysis.calibration import QuantizationCalibrator

from tests.conftest import make_logprobs, make_response_dict


# ── helpers ──────────────────────────────────────────────────────────────


def _make_calibrator(n_reps=3):
    """Return a QuantizationCalibrator with a mocked config."""
    cfg = MagicMock()
    cfg.calibration.local_repetitions = n_reps
    cfg.calibration.expected_kl_ranges = {"q4": [0.0, 0.5], "q8": [0.0, 0.2]}
    cfg.test_params.output_integrity_prompts = {
        "factual": "What is photosynthesis?",
        "reasoning": "Why is the sky blue?",
    }
    return QuantizationCalibrator(cfg)


def _build_results(n_prompts=2, n_reps=3, mean_lp=-1.0, jitter=0.0):
    """Build synthetic calibration results list."""
    results = []
    for p in range(n_prompts):
        responses = []
        for r in range(n_reps):
            lps = make_logprobs([
                (f"tok{i}", mean_lp + jitter * (r - 1) + 0.05 * i)
                for i in range(8)
            ])
            responses.append({"logprobs": lps, "response_text": f"resp_{p}_{r}"})
        results.append({"prompt_name": f"prompt_{p}", "responses": responses})
    return results


# ── TestComputeBaselines ─────────────────────────────────────────────────


class TestComputeBaselines:
    def test_output_structure(self):
        cal = _make_calibrator()
        baselines = cal.compute_baselines(_build_results())
        assert "intra_model_kl" in baselines
        assert "entropy" in baselines
        assert "token_agreement" in baselines
        assert "expected_kl_ranges" in baselines
        for sub in ("mean", "std"):
            assert sub in baselines["intra_model_kl"]
            assert sub in baselines["entropy"]
            assert sub in baselines["token_agreement"]

    def test_near_zero_kl_for_identical_logprobs(self):
        """Identical responses → KL ≈ 0."""
        cal = _make_calibrator()
        baselines = cal.compute_baselines(_build_results(jitter=0.0))
        assert baselines["intra_model_kl"]["mean"] < 0.01

    def test_positive_kl_for_different_logprobs(self):
        """Different responses → KL > 0."""
        cal = _make_calibrator()
        baselines = cal.compute_baselines(_build_results(jitter=1.5))
        assert baselines["intra_model_kl"]["mean"] > 0.0

    def test_perfect_agreement_for_identical_data(self):
        """Identical token sequences → agreement = 1.0."""
        cal = _make_calibrator()
        baselines = cal.compute_baselines(_build_results(jitter=0.0))
        assert baselines["token_agreement"]["mean"] == pytest.approx(1.0)

    def test_empty_logprobs_handled(self):
        """Responses with empty logprobs should not crash."""
        cal = _make_calibrator()
        results = [{"prompt_name": "test", "responses": [
            {"logprobs": [], "response_text": "a"},
            {"logprobs": [], "response_text": "b"},
        ]}]
        baselines = cal.compute_baselines(results)
        assert baselines["intra_model_kl"]["n_pairs"] == 0
        assert baselines["entropy"]["n"] == 0

    def test_none_logprobs_handled(self):
        """Responses with None logprobs should not crash."""
        cal = _make_calibrator()
        results = [{"prompt_name": "test", "responses": [
            {"logprobs": None, "response_text": "a"},
            {"logprobs": None, "response_text": "b"},
        ]}]
        baselines = cal.compute_baselines(results)
        assert baselines["intra_model_kl"]["n_pairs"] == 0

    def test_multi_prompt_aggregation(self):
        """Multiple prompts contribute to aggregated statistics."""
        cal = _make_calibrator()
        results = _build_results(n_prompts=3, n_reps=3)
        baselines = cal.compute_baselines(results)
        # 3 prompts × C(3,2)=3 pairs each = 9 KL pairs
        assert baselines["intra_model_kl"]["n_pairs"] == 9
        # 3 prompts × 3 reps = 9 entropy values
        assert baselines["entropy"]["n"] == 9


# ── TestRunCalibration ───────────────────────────────────────────────────


class TestRunCalibration:
    def test_calls_client_correct_times(self):
        """run_calibration calls chat_completion n_prompts × n_reps times."""
        cal = _make_calibrator(n_reps=3)
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = make_response_dict(mean_lp=-1.0)

        result = cal.run_calibration(mock_client)

        n_prompts = len(cal.config.test_params.output_integrity_prompts)
        assert mock_client.chat_completion.call_count == n_prompts * 3

    def test_output_has_expected_keys(self):
        cal = _make_calibrator(n_reps=2)
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = make_response_dict(mean_lp=-1.0)

        result = cal.run_calibration(mock_client)
        assert "intra_model_kl" in result
        assert "entropy" in result
        assert "token_agreement" in result

    def test_custom_prompts(self):
        """Passing custom prompts overrides config prompts."""
        cal = _make_calibrator(n_reps=2)
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = make_response_dict(mean_lp=-1.0)

        custom = {"only_one": "A single prompt."}
        result = cal.run_calibration(mock_client, prompts=custom)
        assert mock_client.chat_completion.call_count == 2  # 1 prompt × 2 reps
