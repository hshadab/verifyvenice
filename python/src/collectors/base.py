"""Base collector with retry logic and raw data saving."""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import VerifyConfig


class BaseCollector(ABC):
    def __init__(self, config: VerifyConfig):
        self.config = config
        self.results: list[dict[str, Any]] = []

    @abstractmethod
    def collect(self) -> list[dict[str, Any]]:
        ...

    def _generate_test_id(self) -> str:
        return f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _save_raw(self, data: dict[str, Any], provider: str, test_id: str) -> Path:
        raw_dir = self.config.data_dir / "raw" / provider
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{test_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def _retry_with_backoff(
        self,
        fn,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> Any:
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                print(f"  Retry {attempt + 1}/{max_retries} after {delay}s: {e}")
                time.sleep(delay)

    def _rate_limit_sleep(self):
        time.sleep(self.config.test_params.rate_limit_sleep)

    def save_all_results(self, filename: str) -> Path:
        processed_dir = self.config.data_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        path = processed_dir / filename
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        return path
