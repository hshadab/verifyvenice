"""Group A: Output integrity tests.

Compares Venice API responses against reference providers to detect
output filtering or modification.

Venice 70B does not support logprobs — comparison is text-only.
Venice 3B supports per-token logprobs (top-1 only, no top_logprobs).
Together 70B and Ollama 3B support full logprobs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tqdm import tqdm

from src.clients.local import OllamaClient
from src.clients.reference import ReferenceClient
from src.clients.venice import VeniceClient
from src.collectors.base import BaseCollector
from src.config import VerifyConfig


class OutputIntegrityCollector(BaseCollector):
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
        prompts = self.config.test_params.output_integrity_prompts
        seeds = self.config.get_seeds(1 if dry_run else None)
        total = len(prompts) * len(seeds)

        print(f"Group A: Output Integrity — {len(prompts)} prompts x {len(seeds)} seeds = {total} test pairs")
        v70b_has_logprobs = VeniceClient.model_supports_logprobs(self.venice.model_70b)
        v3b_has_logprobs = VeniceClient.model_supports_logprobs(self.venice.model_3b)
        print(f"  Venice 70B ({self.venice.model_70b}) logprobs: {v70b_has_logprobs}")
        print(f"  Venice 3B ({self.venice.model_3b}) logprobs: {v3b_has_logprobs}")

        with tqdm(total=total, desc="Group A") as pbar:
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

                    # Together.ai 70B (full logprobs)
                    together_70b = self._retry_with_backoff(
                        lambda p=prompt_text, s=seed: self.reference.chat_completion(
                            p, seed=s,
                        )
                    )

                    result: dict[str, Any] = {
                        "test_id": test_id,
                        "group": "A",
                        "subgroup": "70b_comparison",
                        "prompt_type": prompt_name,
                        "prompt_text": prompt_text,
                        "seed": seed,
                        "repetition": i,
                        "venice": venice_70b,
                        "reference": together_70b,
                        "comparison_mode": "text_only" if not v70b_has_logprobs else "full",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    # Venice 3B vs Ollama 3B (if Ollama is available)
                    if self.local is not None:
                        venice_3b = self._retry_with_backoff(
                            lambda p=prompt_text, s=seed: self.venice.chat_completion(
                                p, model=self.venice.model_3b, seed=s,
                            )
                        )
                        self._rate_limit_sleep()

                        ollama_3b = self._retry_with_backoff(
                            lambda p=prompt_text, s=seed: self.local.chat_completion(
                                p, seed=s,
                            )
                        )

                        result["venice_3b"] = venice_3b
                        result["local_3b"] = ollama_3b
                        result["3b_comparison_mode"] = "top1_logprob" if v3b_has_logprobs else "text_only"

                    self._save_raw(result, "venice", f"group_a_{test_id}")
                    self.results.append(result)
                    pbar.update(1)

        return self.results

    def collect_system_prompt_differential(self) -> list[dict[str, Any]]:
        """Compare behavior with Venice system prompt on vs off.

        Uses 70B model (text-only comparison since logprobs not supported).
        """
        prompts = self.config.test_params.output_integrity_prompts
        seeds = self.config.get_seeds(3)
        results = []

        print(f"System prompt differential: {len(prompts)} prompts x {len(seeds)} seeds")

        for prompt_name, prompt_text in tqdm(prompts.items(), desc="SysPrompt diff"):
            for seed in seeds:
                test_id = self._generate_test_id()

                with_prompt = self._retry_with_backoff(
                    lambda p=prompt_text, s=seed: self.venice.chat_completion(
                        p, seed=s, include_venice_system_prompt=True,
                    )
                )
                self._rate_limit_sleep()

                without_prompt = self._retry_with_backoff(
                    lambda p=prompt_text, s=seed: self.venice.chat_completion(
                        p, seed=s, include_venice_system_prompt=False,
                    )
                )
                self._rate_limit_sleep()

                result = {
                    "test_id": test_id,
                    "group": "A",
                    "subgroup": "system_prompt_diff",
                    "prompt_type": prompt_name,
                    "prompt_text": prompt_text,
                    "seed": seed,
                    "with_venice_prompt": with_prompt,
                    "without_venice_prompt": without_prompt,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                self._save_raw(result, "venice", f"sysprompt_diff_{test_id}")
                results.append(result)

        return results
