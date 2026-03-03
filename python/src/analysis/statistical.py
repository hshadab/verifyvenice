"""Statistical analysis for logprob distributions.

Core metrics: KL divergence, Shannon entropy, token agreement,
Mann-Whitney U test, bootstrap confidence intervals.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import rel_entr


EPSILON = 1e-10  # smoothing for log-probability distributions


class StatisticalAnalyzer:
    def kl_divergence(
        self,
        p_logprobs: list[dict],
        q_logprobs: list[dict],
    ) -> float:
        """KL(P || Q) between two aligned logprob sequences.

        Each element is a dict with 'top_logprobs': [{'token': str, 'logprob': float}, ...].
        Computes per-token KL divergence and returns the mean.
        """
        if not p_logprobs or not q_logprobs:
            return float("inf")

        min_len = min(len(p_logprobs), len(q_logprobs))
        kl_values = []

        for i in range(min_len):
            p_dist, q_dist = self._align_distributions(
                p_logprobs[i].get("top_logprobs", []),
                q_logprobs[i].get("top_logprobs", []),
            )
            if len(p_dist) > 0 and len(q_dist) > 0:
                kl = float(np.sum(rel_entr(p_dist, q_dist)))
                if np.isfinite(kl):
                    kl_values.append(kl)

        return float(np.mean(kl_values)) if kl_values else float("inf")

    def shannon_entropy(self, logprobs: list[dict]) -> list[float]:
        """Per-token Shannon entropy H(P) = -sum(p * log(p))."""
        entropies = []
        for token_data in logprobs:
            top_lps = token_data.get("top_logprobs", [])
            if not top_lps:
                continue
            probs = np.array([np.exp(t["logprob"]) for t in top_lps])
            probs = probs / probs.sum()  # renormalize over top-k
            probs = np.clip(probs, EPSILON, 1.0)
            h = -float(np.sum(probs * np.log(probs)))
            entropies.append(h)
        return entropies

    def mean_entropy(self, logprobs: list[dict]) -> float:
        entropies = self.shannon_entropy(logprobs)
        return float(np.mean(entropies)) if entropies else 0.0

    def token_agreement_rate(
        self,
        logprobs_a: list[dict],
        logprobs_b: list[dict],
    ) -> float:
        """Fraction of positions where top-1 token matches."""
        if not logprobs_a or not logprobs_b:
            return 0.0

        min_len = min(len(logprobs_a), len(logprobs_b))
        matches = 0

        for i in range(min_len):
            token_a = logprobs_a[i].get("token", "")
            token_b = logprobs_b[i].get("token", "")
            if token_a == token_b:
                matches += 1

        return matches / min_len if min_len > 0 else 0.0

    def entropy_ratio(
        self,
        logprobs_a: list[dict],
        logprobs_b: list[dict],
    ) -> float:
        """Ratio of mean entropies. ~1.0 indicates same model class."""
        h_a = self.mean_entropy(logprobs_a)
        h_b = self.mean_entropy(logprobs_b)
        if h_b < EPSILON:
            return float("inf")
        return h_a / h_b

    def mann_whitney_u(
        self,
        values_a: list[float],
        values_b: list[float],
    ) -> tuple[float, float]:
        """Non-parametric test for distribution difference. Returns (U, p-value)."""
        if len(values_a) < 2 or len(values_b) < 2:
            return 0.0, 1.0
        result = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)

    def bootstrap_confidence_interval(
        self,
        values: list[float],
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> tuple[float, float, float]:
        """Bootstrap CI for a metric. Returns (mean, lower, upper)."""
        arr = np.array(values)
        if len(arr) < 2:
            return float(np.mean(arr)), float(np.mean(arr)), float(np.mean(arr))

        boot_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            boot_means.append(float(np.mean(sample)))

        alpha = (1 - ci) / 2
        lower = float(np.percentile(boot_means, 100 * alpha))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
        return float(np.mean(arr)), lower, upper

    def top1_logprob_stats(self, logprobs: list[dict]) -> dict:
        """Summary statistics for top-1 logprob values."""
        values = [t["logprob"] for t in logprobs if "logprob" in t]
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "n": 0}
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(values),
        }

    def logprob_gap_stats(self, logprobs: list[dict]) -> dict:
        """Gap between top-1 and top-2 logprob per token."""
        gaps = []
        for token_data in logprobs:
            top_lps = token_data.get("top_logprobs", [])
            if len(top_lps) >= 2:
                gap = top_lps[0]["logprob"] - top_lps[1]["logprob"]
                gaps.append(gap)
        if not gaps:
            return {"mean": 0, "std": 0, "n": 0}
        arr = np.array(gaps)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": len(gaps),
        }

    def _align_distributions(
        self,
        p_top: list[dict],
        q_top: list[dict],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align two top-k logprob distributions over the union of tokens.

        Missing tokens get epsilon probability. Both distributions are
        renormalized to sum to 1.
        """
        all_tokens = set()
        p_map: dict[str, float] = {}
        q_map: dict[str, float] = {}

        for t in p_top:
            p_map[t["token"]] = np.exp(t["logprob"])
            all_tokens.add(t["token"])
        for t in q_top:
            q_map[t["token"]] = np.exp(t["logprob"])
            all_tokens.add(t["token"])

        if not all_tokens:
            return np.array([]), np.array([])

        tokens = sorted(all_tokens)
        p_arr = np.array([p_map.get(t, EPSILON) for t in tokens])
        q_arr = np.array([q_map.get(t, EPSILON) for t in tokens])

        # Renormalize
        p_arr = p_arr / p_arr.sum()
        q_arr = q_arr / q_arr.sum()

        return p_arr, q_arr
