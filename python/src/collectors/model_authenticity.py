"""Group B: Model authenticity tests.

Uses logprob distribution fingerprinting to verify Venice serves the
claimed model, not a cheaper substitute.

Venice 70B lacks logprob support, so fingerprinting uses text-based
features for 70B and logprob features for 3B (per-token logprobs).
Together.ai 70B provides full logprobs for reference fingerprinting.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from tqdm import tqdm

from src.clients.local import OllamaClient
from src.clients.reference import ReferenceClient
from src.clients.venice import VeniceClient
from src.collectors.base import BaseCollector
from src.config import VerifyConfig


class ModelAuthenticityCollector(BaseCollector):
    def __init__(
        self,
        config: VerifyConfig,
        venice: VeniceClient,
        reference: ReferenceClient,
        local: OllamaClient | None = None,
    ):
        super().__init__(config)
        self.venice = venice
        self.reference = reference
        self.local = local

    def collect(self, dry_run: bool = False) -> list[dict[str, Any]]:
        prompts = self.config.test_params.model_authenticity_prompts
        seeds = self.config.get_seeds(1 if dry_run else None)
        total = len(prompts) * len(seeds)

        print(f"Group B: Model Authenticity — {len(prompts)} prompts x {len(seeds)} seeds = {total} tests")

        with tqdm(total=total, desc="Group B") as pbar:
            for prompt_name, prompt_text in prompts.items():
                for i, seed in enumerate(seeds):
                    test_id = self._generate_test_id()

                    # Venice 70B (auto-detects logprob support)
                    venice_70b = self._retry_with_backoff(
                        lambda p=prompt_text, s=seed: self.venice.chat_completion(
                            p, model=self.venice.model_70b, seed=s,
                        )
                    )
                    self._rate_limit_sleep()

                    # Venice 3B (per-token logprobs)
                    venice_3b = self._retry_with_backoff(
                        lambda p=prompt_text, s=seed: self.venice.chat_completion(
                            p, model=self.venice.model_3b, seed=s,
                        )
                    )
                    self._rate_limit_sleep()

                    # Together.ai 70B (full logprobs reference)
                    together_70b = self._retry_with_backoff(
                        lambda p=prompt_text, s=seed: self.reference.chat_completion(
                            p, seed=s,
                        )
                    )

                    result: dict[str, Any] = {
                        "test_id": test_id,
                        "group": "B",
                        "prompt_type": prompt_name,
                        "prompt_text": prompt_text,
                        "seed": seed,
                        "repetition": i,
                        "venice_70b": venice_70b,
                        "venice_3b": venice_3b,
                        "together_70b": together_70b,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    # Ollama 3B (full logprobs ground truth)
                    if self.local is not None:
                        ollama_3b = self._retry_with_backoff(
                            lambda p=prompt_text, s=seed: self.local.chat_completion(
                                p, seed=s,
                            )
                        )
                        result["ollama_3b"] = ollama_3b

                    self._save_raw(result, "venice", f"group_b_{test_id}")
                    self.results.append(result)
                    pbar.update(1)

        return self.results

    def collect_temporal_consistency(self, n_rounds: int = 3) -> list[dict[str, Any]]:
        """Test routing consistency: same prompt at different times."""
        prompt = self.config.test_params.model_authenticity_prompts["perplexity_probe"]
        seed = 42
        results = []

        print(f"Temporal consistency: {n_rounds} rounds for routing detection")

        for round_idx in tqdm(range(n_rounds), desc="Temporal"):
            test_id = self._generate_test_id()

            venice_response = self._retry_with_backoff(
                lambda: self.venice.chat_completion(
                    prompt, model=self.venice.model_70b, seed=seed,
                )
            )
            self._rate_limit_sleep()

            result = {
                "test_id": test_id,
                "group": "B",
                "subgroup": "temporal_consistency",
                "prompt_text": prompt,
                "seed": seed,
                "round": round_idx,
                "venice_70b": venice_response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self._save_raw(result, "venice", f"temporal_{test_id}")
            results.append(result)

        return results

    def collect_adversarial(self, n_prompts: int = 5) -> list[dict[str, Any]]:
        """Adversarial robustness: randomized prompts with varied seeds."""
        base_prompts = [
            "Explain how photosynthesis works in simple terms.",
            "What are the main differences between TCP and UDP?",
            "Describe the plot of Romeo and Juliet in three sentences.",
            "How does a binary search algorithm work?",
            "What causes tides in the ocean?",
        ]
        results = []

        print(f"Adversarial probes: {n_prompts} randomized prompts")

        for idx in tqdm(range(n_prompts), desc="Adversarial"):
            test_id = self._generate_test_id()
            prompt = base_prompts[idx % len(base_prompts)]
            seed = random.randint(1, 100000)

            venice_response = self._retry_with_backoff(
                lambda p=prompt, s=seed: self.venice.chat_completion(
                    p, model=self.venice.model_70b, seed=s,
                )
            )
            self._rate_limit_sleep()

            together_response = self._retry_with_backoff(
                lambda p=prompt, s=seed: self.reference.chat_completion(
                    p, seed=s,
                )
            )

            result = {
                "test_id": test_id,
                "group": "B",
                "subgroup": "adversarial",
                "prompt_text": prompt,
                "seed": seed,
                "venice_70b": venice_response,
                "together_70b": together_response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self._save_raw(result, "venice", f"adversarial_{test_id}")
            results.append(result)

        return results
