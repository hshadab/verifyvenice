"""Configuration loader for VerifyVenice.

Loads config.yaml for test parameters and .env for API keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # verifyvenice/


@dataclass
class VeniceConfig:
    base_url: str
    model_70b: str
    model_3b: str


@dataclass
class ReferenceConfig:
    provider: str
    base_url: str
    model_70b: str


@dataclass
class LocalConfig:
    base_url: str
    model_3b: str


@dataclass
class TestParams:
    repetitions: int
    top_logprobs: int
    max_tokens: int
    temperature: float
    seeds: list[int]
    rate_limit_sleep: float
    output_integrity_prompts: dict[str, str]
    model_authenticity_prompts: dict[str, str]


@dataclass
class Thresholds:
    kl_divergence_warn: float
    kl_divergence_fail: float
    token_agreement_min: float
    entropy_ratio_range: list[float]
    cosine_similarity_min: float


@dataclass
class CalibrationConfig:
    local_repetitions: int
    expected_kl_ranges: dict[str, list[float]]


@dataclass
class VerifyConfig:
    venice: VeniceConfig
    reference: ReferenceConfig
    local: LocalConfig
    test_params: TestParams
    thresholds: Thresholds
    calibration: CalibrationConfig
    venice_api_key: str = ""
    together_api_key: str = ""

    @classmethod
    def load(cls, config_path: Path | None = None, env_path: Path | None = None) -> VerifyConfig:
        if config_path is None:
            config_path = PROJECT_ROOT / "config.yaml"
        if env_path is None:
            env_path = PROJECT_ROOT / ".env"

        load_dotenv(env_path)

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        venice_api_key = os.getenv("VENICE_API_KEY", "")
        together_api_key = os.getenv("TOGETHER_API_KEY", "")

        return cls(
            venice=VeniceConfig(**raw["venice"]),
            reference=ReferenceConfig(**raw["reference"]),
            local=LocalConfig(**raw["local"]),
            test_params=TestParams(**raw["test_params"]),
            thresholds=Thresholds(**raw["thresholds"]),
            calibration=CalibrationConfig(**raw["calibration"]),
            venice_api_key=venice_api_key,
            together_api_key=together_api_key,
        )

    def validate(self) -> list[str]:
        errors = []
        if not self.venice_api_key:
            errors.append("VENICE_API_KEY not set in .env")
        if not self.together_api_key:
            errors.append("TOGETHER_API_KEY not set in .env")
        if len(self.test_params.seeds) < self.test_params.repetitions:
            errors.append(
                f"Need at least {self.test_params.repetitions} seeds, "
                f"got {len(self.test_params.seeds)}"
            )
        return errors

    def get_seeds(self, n: int | None = None) -> list[int]:
        n = n or self.test_params.repetitions
        return self.test_params.seeds[:n]

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"
