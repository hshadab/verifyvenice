"""Quantization calibration to establish baseline divergence.

Runs the same prompt multiple times on a local model to measure
nondeterminism, then establishes expected KL divergence ranges.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from src.analysis.statistical import StatisticalAnalyzer
from src.clients.local import OllamaClient
from src.config import VerifyConfig


class QuantizationCalibrator:
    def __init__(self, config: VerifyConfig):
        self.config = config
        self.analyzer = StatisticalAnalyzer()

    def run_calibration(
        self,
        local_client: OllamaClient,
        prompts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run calibration: same prompts repeated to measure nondeterminism.

        Returns baseline statistics for intra-model variance.
        """
        if prompts is None:
            prompts = self.config.test_params.output_integrity_prompts

        n_reps = self.config.calibration.local_repetitions
        all_results: list[dict] = []

        print(f"Calibration: {len(prompts)} prompts x {n_reps} repetitions on local model")

        for prompt_name, prompt_text in tqdm(prompts.items(), desc="Calibrating"):
            responses = []
            for rep in range(n_reps):
                response = local_client.chat_completion(
                    prompt_text,
                    seed=42,  # same seed every time — measures true nondeterminism
                )
                responses.append(response)

            all_results.append({
                "prompt_name": prompt_name,
                "responses": responses,
            })

        return self.compute_baselines(all_results)

    def compute_baselines(self, results: list[dict]) -> dict[str, Any]:
        """Compute baseline KL divergence and entropy statistics."""
        intra_kl_values = []
        entropy_values = []
        agreement_values = []

        for result in results:
            responses = result["responses"]
            logprob_lists = [r["logprobs"] for r in responses if r["logprobs"]]

            # Pairwise KL divergence within same prompt, same seed
            for i in range(len(logprob_lists)):
                for j in range(i + 1, len(logprob_lists)):
                    kl = self.analyzer.kl_divergence(logprob_lists[i], logprob_lists[j])
                    if np.isfinite(kl):
                        intra_kl_values.append(kl)

                    agreement = self.analyzer.token_agreement_rate(
                        logprob_lists[i], logprob_lists[j]
                    )
                    agreement_values.append(agreement)

            # Entropy per response
            for lps in logprob_lists:
                h = self.analyzer.mean_entropy(lps)
                entropy_values.append(h)

        kl_arr = np.array(intra_kl_values) if intra_kl_values else np.array([0.0])
        ent_arr = np.array(entropy_values) if entropy_values else np.array([0.0])
        agr_arr = np.array(agreement_values) if agreement_values else np.array([1.0])

        return {
            "intra_model_kl": {
                "mean": float(np.mean(kl_arr)),
                "std": float(np.std(kl_arr)),
                "p95": float(np.percentile(kl_arr, 95)),
                "max": float(np.max(kl_arr)),
                "n_pairs": len(intra_kl_values),
            },
            "entropy": {
                "mean": float(np.mean(ent_arr)),
                "std": float(np.std(ent_arr)),
                "n": len(entropy_values),
            },
            "token_agreement": {
                "mean": float(np.mean(agr_arr)),
                "std": float(np.std(agr_arr)),
                "min": float(np.min(agr_arr)),
                "n_pairs": len(agreement_values),
            },
            "expected_kl_ranges": self.config.calibration.expected_kl_ranges,
        }
