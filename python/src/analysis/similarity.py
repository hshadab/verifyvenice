"""Text similarity metrics for comparing API responses."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np


class TextSimilarity:
    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """TF-based cosine similarity between two texts."""
        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        all_tokens = set(tokens_a) | set(tokens_b)
        vec_a = np.array([tokens_a.count(t) for t in all_tokens], dtype=float)
        vec_b = np.array([tokens_b.count(t) for t in all_tokens], dtype=float)

        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        return float(dot / (norm_a * norm_b))

    def jaccard_token_similarity(self, text_a: str, text_b: str) -> float:
        """Token-level Jaccard index."""
        set_a = set(self._tokenize(text_a))
        set_b = set(self._tokenize(text_b))

        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def edit_distance_normalized(self, text_a: str, text_b: str) -> float:
        """Levenshtein distance normalized by max length. 0=identical, 1=completely different."""
        if text_a == text_b:
            return 0.0
        if not text_a or not text_b:
            return 1.0

        m, n = len(text_a), len(text_b)
        dp = list(range(n + 1))

        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if text_a[i - 1] == text_b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(dp[j], dp[j - 1], prev)
                prev = temp

        return dp[n] / max(m, n)

    def bleu_1gram(self, reference: str, candidate: str) -> float:
        """Simple unigram BLEU (precision of candidate tokens in reference)."""
        ref_tokens = self._tokenize(reference)
        cand_tokens = self._tokenize(candidate)

        if not cand_tokens:
            return 0.0

        ref_counts = Counter(ref_tokens)
        cand_counts = Counter(cand_tokens)

        clipped = sum(
            min(count, ref_counts.get(token, 0))
            for token, count in cand_counts.items()
        )

        return clipped / len(cand_tokens)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())
