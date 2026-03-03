"""Export trained models to ONNX for JOLT-Atlas circuits.

Usage:
    python scripts/run_onnx_export.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import VerifyConfig
from src.exporters.onnx_export import export_comparison_model, export_fingerprint_model


def main():
    config = VerifyConfig.load()
    models_dir = config.models_dir

    print("=== Exporting ONNX models for JOLT-Atlas ===\n")

    # Circuit 1: Output comparison
    path1 = export_comparison_model(models_dir)
    print()

    # Circuit 2: Model fingerprint classifier
    path2 = export_fingerprint_model(models_dir)
    print()

    print("=== Export complete ===")
    print(f"  Comparison model: {path1}")
    print(f"  Fingerprint model: {path2}")
    print()
    print("Next: Copy these to the Rust circuit workspace:")
    print(f"  cp {path1} rust/output-comparison/models/")
    print(f"  cp {path2} rust/model-fingerprint/models/")


if __name__ == "__main__":
    main()
