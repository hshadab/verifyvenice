"""Tests for report generator (src/report/generator.py)."""

import pytest

from src.report.generator import ReportGenerator


@pytest.fixture
def gen():
    return ReportGenerator()


# ── helpers ──────────────────────────────────────────────────────────────


def _full_analysis_results():
    """Analysis results dict that populates every report section."""
    return {
        "calibration": {
            "intra_model_kl": {"mean": 0.01, "std": 0.005, "p95": 0.02, "max": 0.03, "n_pairs": 9},
            "entropy": {"mean": 1.2, "std": 0.1, "n": 9},
            "token_agreement": {"mean": 0.95, "std": 0.02, "min": 0.9, "n_pairs": 9},
            "expected_kl_ranges": {"q4": [0.0, 0.5]},
        },
        "output_integrity": {
            "70b_comparison": {
                "mode": "text_only",
                "per_prompt": {
                    "factual": {
                        "cosine_similarity": 0.85,
                        "jaccard_similarity": 0.72,
                        "bleu_1gram": 0.68,
                        "edit_distance": 0.35,
                        "n_samples": 3,
                    },
                },
                "mean_cosine_similarity": "0.8500 [0.8000, 0.9000]",
                "n_total": 3,
            },
            "3b_comparison": {
                "mode": "top1_logprob",
                "per_prompt": {
                    "factual": {
                        "token_agreement": 0.45,
                        "cosine_similarity": 0.75,
                        "top1_logprob_correlation": 0.55,
                        "n_samples": 3,
                    },
                },
                "mean_token_agreement": "0.4500 [0.4000, 0.5000]",
                "n_total": 3,
            },
        },
        "model_authenticity": {
            "text_classifier": {
                "cv_accuracy_mean": 0.95,
                "cv_accuracy_std": 0.03,
                "n_samples": 40,
                "n_features": 6,
            },
            "logprob_classifier": {
                "cv_accuracy_mean": 0.92,
                "cv_accuracy_std": 0.04,
                "n_samples": 40,
                "n_features": 4,
            },
            "venice_70b_text_prediction": {
                "predicted_class": "70b-class",
                "confidence": 0.89,
                "verdict": "PASS",
            },
            "ollama_3b_prediction": {
                "predicted_class": "3b-class",
                "confidence": 0.95,
                "verdict": "PASS",
            },
            "feature_importance": {
                "response_length": 0.42,
                "avg_word_length": 0.15,
            },
        },
        "system_prompt_differential": {
            "per_prompt": {
                "factual": {
                    "cosine_similarity": 0.6,
                    "jaccard_similarity": 0.5,
                    "avg_length_with_prompt": 200,
                    "avg_length_without_prompt": 300,
                    "length_ratio": 0.67,
                },
            },
            "mean_cosine_similarity": 0.6,
            "mean_length_ratio": 0.67,
            "identical_responses": "0/3",
            "n_total": 3,
            "finding": "Venice system prompt toggle produces measurably different behavior",
        },
    }


def _proof_results():
    return {
        "similarity_proof": {
            "status": "verified",
            "size_bytes": 1024,
            "verified": True,
        },
    }


# ── TestGenerate ─────────────────────────────────────────────────────────


class TestGenerate:
    def test_returns_string(self, gen):
        report = gen.generate({})
        assert isinstance(report, str)

    def test_contains_all_section_headers(self, gen):
        report = gen.generate(_full_analysis_results(), _proof_results())
        for header in [
            "# VerifyVenice",
            "## Executive Summary",
            "## Methodology",
            "## Calibration",
            "## Group A",
            "## Group B",
            "## System Prompt",
            "## zkML Proof",
            "## Limitations",
            "## Independence",
        ]:
            assert header in report, f"Missing section: {header}"

    def test_sections_separated_by_hr(self, gen):
        report = gen.generate(_full_analysis_results())
        assert "---" in report


# ── TestExecutiveSummary ─────────────────────────────────────────────────


class TestExecutiveSummary:
    def test_renders_70b_data(self, gen):
        section = gen._executive_summary(_full_analysis_results())
        assert "70B Output Integrity" in section

    def test_renders_3b_data(self, gen):
        section = gen._executive_summary(_full_analysis_results())
        assert "3B Output Integrity" in section

    def test_renders_model_authenticity(self, gen):
        section = gen._executive_summary(_full_analysis_results())
        assert "70B Model Authenticity" in section

    def test_empty_results(self, gen):
        section = gen._executive_summary({})
        assert "Executive Summary" in section


# ── TestCalibrationSection ───────────────────────────────────────────────


class TestCalibrationSection:
    def test_not_yet_run_when_empty(self, gen):
        section = gen._calibration_results({})
        assert "not yet run" in section.lower()

    def test_renders_table_when_populated(self, gen):
        cal = _full_analysis_results()["calibration"]
        section = gen._calibration_results(cal)
        assert "Intra-model KL" in section
        assert "Mean entropy" in section
        assert "Token agreement" in section


# ── TestGroupAResults ────────────────────────────────────────────────────


class TestGroupAResults:
    def test_empty_state(self, gen):
        section = gen._group_a_results({})
        assert "not yet run" in section.lower()

    def test_70b_text_table_with_verdicts(self, gen):
        oi = _full_analysis_results()["output_integrity"]
        section = gen._group_a_results(oi)
        assert "70B" in section
        assert "Cosine Sim" in section
        assert "Verdict" in section

    def test_3b_logprob_table(self, gen):
        oi = _full_analysis_results()["output_integrity"]
        section = gen._group_a_results(oi)
        assert "3B" in section
        assert "Token Agreement" in section


# ── TestGroupBResults ────────────────────────────────────────────────────


class TestGroupBResults:
    def test_empty_state(self, gen):
        section = gen._group_b_results({})
        assert "not yet run" in section.lower()

    def test_classifier_accuracy(self, gen):
        ma = _full_analysis_results()["model_authenticity"]
        section = gen._group_b_results(ma)
        assert "CV Accuracy" in section

    def test_prediction_table(self, gen):
        ma = _full_analysis_results()["model_authenticity"]
        section = gen._group_b_results(ma)
        assert "Venice 70B" in section
        assert "70b-class" in section


# ── TestSystemPromptResults ──────────────────────────────────────────────


class TestSystemPromptResults:
    def test_empty_state(self, gen):
        section = gen._system_prompt_results({})
        assert "not yet run" in section.lower()

    def test_finding_text(self, gen):
        sp = _full_analysis_results()["system_prompt_differential"]
        section = gen._system_prompt_results(sp)
        assert "measurably different behavior" in section

    def test_per_prompt_table(self, gen):
        sp = _full_analysis_results()["system_prompt_differential"]
        section = gen._system_prompt_results(sp)
        assert "factual" in section
        assert "Cosine Sim" in section


# ── TestTextVerdict ──────────────────────────────────────────────────────


class TestTextVerdict:
    def test_pass(self, gen):
        assert gen._text_verdict(0.9, 0.9) == "PASS"

    def test_warn(self, gen):
        assert gen._text_verdict(0.5, 0.5) == "WARN"

    def test_fail(self, gen):
        assert gen._text_verdict(0.1, 0.1) == "FAIL"

    def test_boundary_pass(self, gen):
        # avg = 0.71 > 0.7 → PASS
        assert gen._text_verdict(0.71, 0.71) == "PASS"

    def test_boundary_warn(self, gen):
        # avg = 0.41 > 0.4 → WARN
        assert gen._text_verdict(0.41, 0.41) == "WARN"

    def test_boundary_fail(self, gen):
        # avg = 0.4 → not > 0.4 → FAIL
        assert gen._text_verdict(0.4, 0.4) == "FAIL"

    def test_configurable_thresholds(self, gen):
        """Validates fix #3: thresholds are class attributes and can be overridden."""
        gen.TEXT_VERDICT_PASS_THRESHOLD = 0.9
        gen.TEXT_VERDICT_WARN_THRESHOLD = 0.6
        # avg = 0.75 — was PASS with default 0.7, now WARN
        assert gen._text_verdict(0.75, 0.75) == "WARN"
        # avg = 0.65 — was WARN with default 0.4, still WARN with 0.6
        assert gen._text_verdict(0.65, 0.65) == "WARN"
        # avg = 0.5 — was WARN with default 0.4, now FAIL with 0.6
        assert gen._text_verdict(0.5, 0.5) == "FAIL"


# ── TestSave ─────────────────────────────────────────────────────────────


class TestSave:
    def test_creates_md_file(self, gen, tmp_path):
        report = gen.generate({})
        path = gen.save(report, tmp_path)
        assert path.exists()
        assert path.suffix == ".md"
        assert path.read_text() == report

    def test_creates_directory(self, gen, tmp_path):
        sub = tmp_path / "nested" / "dir"
        report = gen.generate({})
        path = gen.save(report, sub)
        assert path.exists()


# ── TestZkmlSection ──────────────────────────────────────────────────────


class TestZkmlSection:
    def test_not_yet_generated_when_empty(self, gen):
        section = gen._zkml_proof_summary({})
        assert "not yet generated" in section.lower()

    def test_renders_proof_table(self, gen):
        section = gen._zkml_proof_summary(_proof_results())
        assert "similarity_proof" in section
        assert "verified" in section
        assert "1024" in section
