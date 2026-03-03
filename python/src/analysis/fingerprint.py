"""Model fingerprinting via logprob distribution and text features.

Extracts feature vectors from responses and trains a classifier to
distinguish model families. Adapts to available data:
- Full logprobs (8 features): Together.ai, Ollama
- Top-1 logprobs only (4 features): Venice 3B (vLLM)
- Text-only (6 features): Venice 70B (no logprob support)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class ModelFingerprinter:
    # Full 8-dim features (requires top_logprobs)
    FULL_FEATURE_NAMES = [
        "mean_top1_logprob",
        "std_top1_logprob",
        "mean_entropy",
        "mean_top1_top2_gap",
        "frac_high_confidence",
        "mean_perplexity",
        "vocab_coverage",
        "mean_top5_spread",
    ]

    # Top-1 only features (4 features, no top_logprobs needed)
    TOP1_FEATURE_NAMES = [
        "mean_top1_logprob",
        "std_top1_logprob",
        "frac_high_confidence",
        "mean_perplexity",
    ]

    # Text-based features (no logprobs needed at all)
    TEXT_FEATURE_NAMES = [
        "response_length",
        "avg_word_length",
        "type_token_ratio",
        "sentence_count",
        "avg_sentence_length",
        "punctuation_ratio",
    ]

    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self._is_fitted = False
        self._feature_names: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names or self.FULL_FEATURE_NAMES

    def extract_features(self, response: dict) -> np.ndarray:
        """Extract 8-dim feature vector from response with full logprobs."""
        logprobs = response.get("logprobs", [])
        if not logprobs:
            return np.zeros(8)

        top1_logprobs = []
        entropies = []
        gaps = []
        high_conf_count = 0
        all_top_tokens: set[str] = set()
        spreads = []

        for token_data in logprobs:
            top_lps = token_data.get("top_logprobs", [])
            top1_lp = token_data.get("logprob", top_lps[0].get("logprob", 0) if top_lps else 0)
            top1_logprobs.append(top1_lp)

            if top_lps:
                probs = np.array([np.exp(t["logprob"]) for t in top_lps])
                probs = probs / probs.sum()
                probs = np.clip(probs, 1e-10, 1.0)
                h = -float(np.sum(probs * np.log(probs)))
                entropies.append(h)

                if len(top_lps) >= 2:
                    gaps.append(top_lps[0]["logprob"] - top_lps[1]["logprob"])

                if np.exp(top_lps[0]["logprob"]) > 0.9:
                    high_conf_count += 1

                for t in top_lps:
                    all_top_tokens.add(t["token"])

                if len(top_lps) >= 2:
                    lp_vals = [t["logprob"] for t in top_lps]
                    spreads.append(float(np.std(lp_vals)))
            else:
                # Top-1 only: use per-token logprob for confidence
                if np.exp(top1_lp) > 0.9:
                    high_conf_count += 1

        n_tokens = len(logprobs)
        top1_arr = np.array(top1_logprobs) if top1_logprobs else np.array([0.0])

        return np.array([
            float(np.mean(top1_arr)),
            float(np.std(top1_arr)),
            float(np.mean(entropies)) if entropies else 0.0,
            float(np.mean(gaps)) if gaps else 0.0,
            high_conf_count / n_tokens if n_tokens > 0 else 0.0,
            float(np.exp(-np.mean(top1_arr))),
            len(all_top_tokens),
            float(np.mean(spreads)) if spreads else 0.0,
        ])

    def extract_top1_features(self, response: dict) -> np.ndarray:
        """Extract 4-dim features using only per-token logprobs (no top_logprobs)."""
        logprobs = response.get("logprobs", [])
        if not logprobs:
            return np.zeros(4)

        top1_logprobs = []
        high_conf_count = 0

        for token_data in logprobs:
            lp = token_data.get("logprob", 0)
            top1_logprobs.append(lp)
            if np.exp(lp) > 0.9:
                high_conf_count += 1

        n_tokens = len(logprobs)
        top1_arr = np.array(top1_logprobs) if top1_logprobs else np.array([0.0])

        return np.array([
            float(np.mean(top1_arr)),
            float(np.std(top1_arr)),
            high_conf_count / n_tokens if n_tokens > 0 else 0.0,
            float(np.exp(-np.mean(top1_arr))),
        ])

    def extract_text_features(self, response: dict) -> np.ndarray:
        """Extract 6-dim text-based features (no logprobs needed)."""
        text = response.get("response_text", "")
        if not text:
            return np.zeros(6)

        words = re.findall(r'\w+', text.lower())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        n_words = len(words)
        n_chars = sum(len(w) for w in words)
        unique_words = len(set(words))
        n_sentences = max(len(sentences), 1)
        n_punct = sum(1 for c in text if c in '.,;:!?()-"\'')

        return np.array([
            n_words,                                              # response_length
            n_chars / n_words if n_words > 0 else 0.0,           # avg_word_length
            unique_words / n_words if n_words > 0 else 0.0,      # type_token_ratio
            n_sentences,                                           # sentence_count
            n_words / n_sentences,                                 # avg_sentence_length
            n_punct / len(text) if text else 0.0,                 # punctuation_ratio
        ])

    def build_feature_matrix(self, responses: list[dict]) -> np.ndarray:
        """Stack full 8-dim features. Shape: (n_responses, 8)."""
        features = [self.extract_features(r) for r in responses]
        return np.array(features)

    def build_top1_feature_matrix(self, responses: list[dict]) -> np.ndarray:
        """Stack top-1 only features. Shape: (n_responses, 4)."""
        return np.array([self.extract_top1_features(r) for r in responses])

    def build_text_feature_matrix(self, responses: list[dict]) -> np.ndarray:
        """Stack text features. Shape: (n_responses, 6)."""
        return np.array([self.extract_text_features(r) for r in responses])

    def train_classifier(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Train logistic regression to distinguish model classes.

        Labels: 0=70b-class, 1=3b-class
        """
        self._feature_names = feature_names or self.FULL_FEATURE_NAMES
        X = self.scaler.fit_transform(features)
        self.classifier.fit(X, labels)
        self._is_fitted = True

        cv_scores = cross_val_score(self.classifier, X, labels, cv=min(5, len(labels)))

        return {
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "n_samples": len(labels),
            "n_features": features.shape[1],
            "feature_names": self._feature_names,
            "classes": list(np.unique(labels).tolist()),
        }

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict model class and confidence."""
        if not self._is_fitted:
            raise RuntimeError("Classifier not trained yet. Call train_classifier first.")
        X = self.scaler.transform(features)
        labels = self.classifier.predict(X)
        probs = self.classifier.predict_proba(X)
        return labels, probs

    def get_feature_importance(self) -> dict[str, float]:
        if not self._is_fitted:
            return {}
        coefs = np.abs(self.classifier.coef_).mean(axis=0)
        names = self._feature_names or self.FULL_FEATURE_NAMES
        return dict(zip(names, coefs.tolist()))
