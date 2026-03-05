"""Integration tests for analysis pipeline (scripts/run_analysis.py)."""

import numpy as np
import pytest

from scripts.run_analysis import (
    analyze_model_authenticity,
    analyze_output_integrity,
    analyze_system_prompt_diff,
)
from src.analysis.fingerprint import ModelFingerprinter
from src.analysis.similarity import TextSimilarity
from src.analysis.statistical import StatisticalAnalyzer
from src.report.generator import ReportGenerator

from tests.conftest import make_logprobs, make_response_dict


# ── data builders ────────────────────────────────────────────────────────


def _build_70b_response(prompt_text="What is X?"):
    """70B-style: text only, empty logprobs."""
    return make_response_dict(
        response_text=f"Detailed answer about {prompt_text[:20]}. "
                      "This is a longer explanation covering multiple aspects.",
        logprobs=[],
        n_tokens=0,
    )


def _build_3b_response(prompt_text="What is X?"):
    """3B-style: shorter text, logprobs with mean_lp=-1.5."""
    return make_response_dict(
        response_text=f"Short answer about {prompt_text[:20]}.",
        mean_lp=-1.5,
        n_tokens=10,
    )


def _build_together_70b_response(prompt_text="What is X?"):
    """Together 70B: text + full top-5 logprobs, mean_lp=-0.3."""
    tokens = [f"word{i}" for i in range(15)]
    lps = []
    for i, token in enumerate(tokens):
        base_lp = -0.3 + (i % 3) * 0.05
        top5 = [
            {"token": token, "logprob": base_lp},
            {"token": f"alt1_{token}", "logprob": base_lp - 1.0},
            {"token": f"alt2_{token}", "logprob": base_lp - 2.0},
            {"token": f"alt3_{token}", "logprob": base_lp - 3.0},
            {"token": f"alt4_{token}", "logprob": base_lp - 4.0},
        ]
        lps.append({
            "token": token,
            "logprob": base_lp,
            "top_logprobs": top5,
        })
    return {
        "response_text": f"Detailed reference answer about {prompt_text[:20]}. "
                         "Thorough explanation with many details and examples.",
        "logprobs": lps,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 20, "completion_tokens": 15},
        "hashes": {},
        "latency": 0.4,
    }


def _build_group_a_results(n_per_prompt=3):
    """Build Group A (output integrity) results list."""
    prompts = {"factual": "What is photosynthesis?", "reasoning": "Why is the sky blue?"}
    results = []
    for pname, ptext in prompts.items():
        for i in range(n_per_prompt):
            results.append({
                "prompt_type": pname,
                "venice": _build_70b_response(ptext),
                "reference": _build_together_70b_response(ptext),
                "venice_3b": _build_3b_response(ptext),
                "local_3b": _build_3b_response(ptext),
            })
    return results


def _build_group_b_results(n_per_class=5):
    """Build Group B (model authenticity) results list."""
    results = []
    for i in range(n_per_class):
        results.append({
            "venice_70b": {
                "response_text": f"Venice 70B response {i}. Detailed and thorough explanation "
                                 "covering many aspects of the topic with nuance.",
                "logprobs": [],
            },
            "venice_3b": _build_3b_response(f"Venice 3B prompt {i}"),
            "together_70b": _build_together_70b_response(f"Together prompt {i}"),
            "ollama_3b": _build_3b_response(f"Ollama prompt {i}"),
        })
    return results


def _build_sysprompt_results(n=3):
    """Build system prompt differential results."""
    results = []
    prompts = ["factual", "reasoning"]
    for pname in prompts:
        for i in range(n):
            results.append({
                "prompt_type": pname,
                "with_venice_prompt": {
                    "response_text": f"Short response with system prompt for {pname} {i}.",
                },
                "without_venice_prompt": {
                    "response_text": f"Longer response without the Venice system prompt. "
                                     f"This version for {pname} {i} tends to be more verbose "
                                     "and contains additional details.",
                },
            })
    return results


# ── TestAnalyzeOutputIntegrity ───────────────────────────────────────────


class TestAnalyzeOutputIntegrity:
    @pytest.fixture
    def pipeline_deps(self):
        return StatisticalAnalyzer(), TextSimilarity()

    @pytest.fixture
    def mock_config(self):
        from unittest.mock import MagicMock
        cfg = MagicMock()
        cfg.thresholds.cosine_similarity_min = 0.85
        return cfg

    def test_returns_70b_comparison(self, pipeline_deps, mock_config):
        analyzer, sim = pipeline_deps
        result = analyze_output_integrity(_build_group_a_results(), analyzer, sim, mock_config)
        assert "70b_comparison" in result
        assert result["70b_comparison"]["mode"] == "text_only"

    def test_returns_3b_comparison(self, pipeline_deps, mock_config):
        analyzer, sim = pipeline_deps
        result = analyze_output_integrity(_build_group_a_results(), analyzer, sim, mock_config)
        assert "3b_comparison" in result

    def test_per_prompt_structure(self, pipeline_deps, mock_config):
        analyzer, sim = pipeline_deps
        result = analyze_output_integrity(_build_group_a_results(), analyzer, sim, mock_config)
        per_prompt = result["70b_comparison"]["per_prompt"]
        assert "factual" in per_prompt
        assert "reasoning" in per_prompt
        for metrics in per_prompt.values():
            assert "cosine_similarity" in metrics
            assert "jaccard_similarity" in metrics
            assert "edit_distance" in metrics
            assert "bleu_1gram" in metrics

    def test_cosine_range(self, pipeline_deps, mock_config):
        analyzer, sim = pipeline_deps
        result = analyze_output_integrity(_build_group_a_results(), analyzer, sim, mock_config)
        for metrics in result["70b_comparison"]["per_prompt"].values():
            assert 0.0 <= metrics["cosine_similarity"] <= 1.0

    def test_empty_input(self, pipeline_deps, mock_config):
        analyzer, sim = pipeline_deps
        result = analyze_output_integrity([], analyzer, sim, mock_config)
        assert result["70b_comparison"]["n_total"] == 0

    def test_n_total_matches(self, pipeline_deps, mock_config):
        analyzer, sim = pipeline_deps
        data = _build_group_a_results(n_per_prompt=4)
        result = analyze_output_integrity(data, analyzer, sim, mock_config)
        assert result["70b_comparison"]["n_total"] == 8  # 2 prompts × 4


# ── TestAnalyzeModelAuthenticity ─────────────────────────────────────────


class TestAnalyzeModelAuthenticity:
    def test_text_classifier_trained(self):
        fp = ModelFingerprinter()
        result = analyze_model_authenticity(_build_group_b_results(n_per_class=5), fp)
        assert "text_classifier" in result
        assert result["text_classifier"]["n_samples"] > 0

    def test_venice_70b_prediction_produced(self):
        fp = ModelFingerprinter()
        result = analyze_model_authenticity(_build_group_b_results(n_per_class=5), fp)
        assert "venice_70b_text_prediction" in result
        pred = result["venice_70b_text_prediction"]
        assert "predicted_class" in pred
        assert "confidence" in pred
        assert "verdict" in pred

    def test_ollama_validation(self):
        fp = ModelFingerprinter()
        result = analyze_model_authenticity(_build_group_b_results(n_per_class=5), fp)
        if "ollama_3b_prediction" in result:
            pred = result["ollama_3b_prediction"]
            assert pred["expected_class"] == "3b-class"
            assert "verdict" in pred

    def test_feature_summary(self):
        fp = ModelFingerprinter()
        result = analyze_model_authenticity(_build_group_b_results(n_per_class=5), fp)
        assert "feature_summary" in result
        assert len(result["feature_summary"]) > 0

    def test_minimum_samples_works(self):
        """Need at least 5 per class for cv=5 (sklearn StratifiedKFold constraint)."""
        fp = ModelFingerprinter()
        result = analyze_model_authenticity(_build_group_b_results(n_per_class=5), fp)
        assert "text_classifier" in result

    def test_less_than_2_per_class_skips(self):
        """With <2 per class the classifier cannot train."""
        fp = ModelFingerprinter()
        data = _build_group_b_results(n_per_class=1)
        result = analyze_model_authenticity(data, fp)
        # With only 1 sample per class, can't train (need >=2 per class && >=4 total)
        # The text classifier requires len(t70b_text) >= 2 and len(v3b_text) >= 2
        # With 1 per class, both have exactly 1, so classifier should be skipped
        assert "text_classifier" not in result

    def test_logprob_classifier_trained(self):
        fp = ModelFingerprinter()
        result = analyze_model_authenticity(_build_group_b_results(n_per_class=5), fp)
        # Together 70B has full logprobs, Venice 3B has logprobs → classifier trains
        assert "logprob_classifier" in result


# ── TestAnalyzeSystemPromptDiff ──────────────────────────────────────────


class TestAnalyzeSystemPromptDiff:
    def test_per_prompt_keys(self):
        sim = TextSimilarity()
        result = analyze_system_prompt_diff(_build_sysprompt_results(), sim)
        assert "per_prompt" in result
        assert "factual" in result["per_prompt"]
        assert "reasoning" in result["per_prompt"]

    def test_non_identical_detection(self):
        sim = TextSimilarity()
        result = analyze_system_prompt_diff(_build_sysprompt_results(), sim)
        assert result["identical_responses"] == "0/6"
        assert "measurably different" in result["finding"]

    def test_cosine_range(self):
        sim = TextSimilarity()
        result = analyze_system_prompt_diff(_build_sysprompt_results(), sim)
        for metrics in result["per_prompt"].values():
            assert 0.0 <= metrics["cosine_similarity"] <= 1.0

    def test_empty_input(self):
        sim = TextSimilarity()
        result = analyze_system_prompt_diff([], sim)
        assert result["n_total"] == 0
        assert result["mean_cosine_similarity"] == 0.0

    def test_length_ratio(self):
        sim = TextSimilarity()
        result = analyze_system_prompt_diff(_build_sysprompt_results(), sim)
        # "with" text is shorter than "without" → ratio < 1.0
        for metrics in result["per_prompt"].values():
            assert metrics["length_ratio"] > 0.0


# ── TestPipelineEndToEnd ─────────────────────────────────────────────────


class TestPipelineEndToEnd:
    def test_all_analyses_feed_into_report(self):
        analyzer = StatisticalAnalyzer()
        sim = TextSimilarity()
        fp = ModelFingerprinter()

        from unittest.mock import MagicMock
        cfg = MagicMock()
        cfg.thresholds.cosine_similarity_min = 0.85

        analysis = {
            "output_integrity": analyze_output_integrity(
                _build_group_a_results(), analyzer, sim, cfg
            ),
            "model_authenticity": analyze_model_authenticity(
                _build_group_b_results(n_per_class=5), fp
            ),
            "system_prompt_differential": analyze_system_prompt_diff(
                _build_sysprompt_results(), sim
            ),
        }

        gen = ReportGenerator()
        report = gen.generate(analysis)

        assert len(report) > 500
        assert "Group A" in report
        assert "Group B" in report
        assert "System Prompt" in report
        assert "Executive Summary" in report
