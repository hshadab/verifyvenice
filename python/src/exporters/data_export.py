"""Data export utilities for saving results in various formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


class DataExporter:
    def save_json(self, data: Any, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def save_csv(self, records: list[dict], path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            path.touch()
            return path

        fieldnames = list(records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                # Flatten complex values to strings
                flat = {
                    k: json.dumps(v) if isinstance(v, (dict, list)) else v
                    for k, v in record.items()
                }
                writer.writerow(flat)
        return path

    def save_numpy(self, array: np.ndarray, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), array)
        return path

    def save_feature_matrix(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: list[str],
        path: str | Path,
    ) -> Path:
        """Save feature matrix with labels and column names."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            features=features,
            labels=labels,
            feature_names=np.array(feature_names),
        )
        return path

    def load_feature_matrix(self, path: str | Path) -> dict:
        data = np.load(str(path), allow_pickle=True)
        return {
            "features": data["features"],
            "labels": data["labels"],
            "feature_names": list(data["feature_names"]),
        }
