"""Venice.ai API client with logprob capture.

Venice logprob support varies by model. Models served via vLLM (e.g.
llama-3.2-3b) return per-token logprobs but NOT top_logprobs alternatives.
Some models (e.g. llama-3.3-70b) don't support logprobs at all.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from openai import OpenAI

from src.config import VerifyConfig

# Models that support logprobs (from Venice /models endpoint, supportsLogProbs).
# vLLM-hosted models return per-token logprob but empty top_logprobs.
LOGPROB_MODELS = {
    "venice-uncensored", "qwen3-4b", "llama-3.2-3b",
    "qwen3-235b-a22b-thinking-2507", "qwen3-235b-a22b-instruct-2507",
    "qwen3-next-80b", "qwen3-coder-480b-a35b-instruct",
    "qwen3-5-35b-a3b", "qwen3-vl-235b-a22b",
    "qwen3-coder-480b-a35b-instruct-turbo",
    "kimi-k2-thinking", "grok-41-fast", "grok-code-fast-1",
    "gemini-3-pro-preview", "gemini-3-1-pro-preview",
    "gemini-3-flash-preview", "openai-gpt-52", "openai-gpt-52-codex",
    "openai-gpt-53-codex", "openai-gpt-4o-2024-11-20",
    "openai-gpt-4o-mini-2024-07-18", "minimax-m21",
}


class VeniceClient:
    def __init__(self, config: VerifyConfig):
        self.client = OpenAI(
            api_key=config.venice_api_key,
            base_url=config.venice.base_url,
        )
        self.model_70b = config.venice.model_70b
        self.model_3b = config.venice.model_3b
        self.top_logprobs = config.test_params.top_logprobs
        self.max_tokens = config.test_params.max_tokens
        self.temperature = config.test_params.temperature
        self.rate_limit_sleep = config.test_params.rate_limit_sleep

    @staticmethod
    def model_supports_logprobs(model: str) -> bool:
        return model in LOGPROB_MODELS

    def chat_completion(
        self,
        prompt: str,
        model: str | None = None,
        seed: int = 42,
        top_logprobs: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        include_venice_system_prompt: bool = False,
    ) -> dict[str, Any]:
        model = model or self.model_70b
        top_logprobs = top_logprobs if top_logprobs is not None else self.top_logprobs
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        request_logprobs = self.model_supports_logprobs(model)

        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        start_time = time.monotonic()

        create_kwargs: dict[str, Any] = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            extra_body={
                "venice_parameters": {
                    "include_venice_system_prompt": include_venice_system_prompt,
                }
            },
        )
        if request_logprobs:
            create_kwargs["logprobs"] = True
            create_kwargs["top_logprobs"] = top_logprobs

        response = self.client.chat.completions.create(**create_kwargs)

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
            "provider": "venice",
            "model_requested": model,
            "model_returned": response.model,
            "prompt_hash": prompt_hash,
            "response_text": response_text,
            "response_hash": response_hash,
            "logprobs": logprobs_data,
            "logprobs_supported": request_logprobs,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            "latency_seconds": latency,
            "seed": seed,
            "temperature": temperature,
            "top_logprobs_requested": top_logprobs if request_logprobs else 0,
            "include_venice_system_prompt": include_venice_system_prompt,
        }

    def list_models(self) -> list[dict]:
        models = self.client.models.list()
        return [{"id": m.id, "owned_by": m.owned_by} for m in models.data]
