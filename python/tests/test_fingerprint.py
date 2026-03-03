"""Tests for model fingerprinting module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.fingerprint import ModelFingerprinter


@pytest.fixture
def fingerprinter():
    return ModelFingerprinter()


def _make_response(mean_logprob: float, entropy_level: float) -> dict:
    """Create a synthetic response with controlled logprob characteristics."""
    n_tokens = 20
    logprobs = []
    for _ in range(n_tokens):
        base_lp = mean_logprob + np.random.normal(0, 0.3)
        top_lps = [
            {"token": f"t{i}", "logprob": base_lp - i * entropy_level}
            for i in range(5)
        ]
        logprobs.append({
            "token": top_lps[0]["token"],
            "logprob": top_lps[0]["logprob"],
            "top_logprobs": top_lps,
        })
    return {"logprobs": logprobs}


class TestFeatureExtraction:
    def test_feature_dimensions(self, fingerprinter):
        response = _make_response(-0.5, 0.5)
        features = fingerprinter.extract_features(response)
        assert features.shape == (8,)

    def test_empty_logprobs(self, fingerprinter):
        features = fingerprinter.extract_features({"logprobs": []})
        assert features.shape == (8,)
        assert np.all(features == 0)

    def test_different_models_produce_different_features(self, fingerprinter):
        # 70B-like: high confidence, low entropy
        response_70b = _make_response(-0.2, 1.5)
        # 3B-like: lower confidence, higher entropy
        response_3b = _make_response(-1.5, 0.3)

        feat_70b = fingerprinter.extract_features(response_70b)
        feat_3b = fingerprinter.extract_features(response_3b)

        # Mean top-1 logprob should differ
        assert feat_70b[0] != feat_3b[0]
        # Mean entropy should differ
        assert feat_70b[2] != feat_3b[2]

    def test_build_feature_matrix(self, fingerprinter):
        responses = [_make_response(-0.5, 0.5) for _ in range(10)]
        matrix = fingerprinter.build_feature_matrix(responses)
        assert matrix.shape == (10, 8)


class TestClassifier:
    def test_train_and_predict(self, fingerprinter):
        np.random.seed(42)

        # Generate separable classes
        responses_70b = [_make_response(-0.2, 1.5) for _ in range(20)]
        responses_3b = [_make_response(-1.5, 0.3) for _ in range(20)]

        feat_70b = fingerprinter.build_feature_matrix(responses_70b)
        feat_3b = fingerprinter.build_feature_matrix(responses_3b)

        features = np.vstack([feat_70b, feat_3b])
        labels = np.array([0] * 20 + [1] * 20)

        metrics = fingerprinter.train_classifier(features, labels)
        assert metrics["n_samples"] == 40
        assert metrics["cv_accuracy_mean"] > 0.5  # should be well above chance

    def test_predict_before_training_raises(self, fingerprinter):
        features = np.random.randn(5, 8)
        with pytest.raises(RuntimeError, match="not trained"):
            fingerprinter.predict(features)

    def test_feature_importance(self, fingerprinter):
        np.random.seed(42)
        responses_70b = [_make_response(-0.2, 1.5) for _ in range(20)]
        responses_3b = [_make_response(-1.5, 0.3) for _ in range(20)]

        features = np.vstack([
            fingerprinter.build_feature_matrix(responses_70b),
            fingerprinter.build_feature_matrix(responses_3b),
        ])
        labels = np.array([0] * 20 + [1] * 20)
        fingerprinter.train_classifier(features, labels)

        importance = fingerprinter.get_feature_importance()
        assert len(importance) == 8
        assert all(v >= 0 for v in importance.values())
