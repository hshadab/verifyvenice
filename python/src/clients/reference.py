"""Together.ai reference client for 70B model baseline."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from openai import OpenAI

from src.config import VerifyConfig


class ReferenceClient:
    def __init__(self, config: VerifyConfig):
        self.client = OpenAI(
            api_key=config.together_api_key,
            base_url=config.reference.base_url,
        )
        self.model_70b = config.reference.model_70b
        self.provider = config.reference.provider
        self.top_logprobs = config.test_params.top_logprobs
        self.max_tokens = config.test_params.max_tokens
        self.temperature = config.test_params.temperature

    def chat_completion(
        self,
        prompt: str,
        seed: int = 42,
        top_logprobs: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        top_logprobs = top_logprobs if top_logprobs is not None else self.top_logprobs
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        start_time = time.monotonic()

        response = self.client.chat.completions.create(
            model=self.model_70b,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            seed=seed,
            logprobs=True,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
        )

        latency = time.monotonic() - start_time
        choice = response.choices[0]

        logprobs_data = []
        if choice.logprobs and choice.logprobs.content:
            for token_lp in choice.logprobs.content:
                logprobs_data.append({
                    "token": token_lp.token,
                    "logprob": token_lp.logprob,
                    "top_logprobs": [
                        {"token": t.token, "logprob": t.logprob}
                        for t in (token_lp.top_logprobs or [])
                    ],
                })

        response_text = choice.message.content or ""
        response_hash = hashlib.sha256(response_text.encode()).hexdigest()

        return {
            "provider": self.provider,
            "model_requested": self.model_70b,
            "model_returned": response.model,
            "prompt_hash": prompt_hash,
            "response_text": response_text,
            "response_hash": response_hash,
            "logprobs": logprobs_data,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            "latency_seconds": latency,
            "seed": seed,
            "temperature": temperature,
            "top_logprobs_requested": top_logprobs,
        }
