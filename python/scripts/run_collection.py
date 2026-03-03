"""Main data collection script.

Usage:
    python scripts/run_collection.py [--dry-run] [--group a|b|all] [--skip-calibration]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.calibration import QuantizationCalibrator
from src.clients.local import OllamaClient
from src.clients.reference import ReferenceClient
from src.clients.venice import VeniceClient
from src.collectors.model_authenticity import ModelAuthenticityCollector
from src.collectors.output_integrity import OutputIntegrityCollector
from src.config import VerifyConfig
from src.exporters.data_export import DataExporter


def estimate_cost(config: VerifyConfig) -> dict:
    n_prompts_a = len(config.test_params.output_integrity_prompts)
    n_prompts_b = len(config.test_params.model_authenticity_prompts)
    n_seeds = config.test_params.repetitions
    max_tokens = config.test_params.max_tokens

    # Venice costs: ~$0.40/M input, ~$1.00/M output
    tokens_per_call = 50 + max_tokens  # rough estimate
    cost_per_call = tokens_per_call * 1.0 / 1_000_000  # output-dominated

    group_a_calls = n_prompts_a * n_seeds * 2  # Venice 70B + 3B
    group_b_calls = n_prompts_b * n_seeds * 2  # Venice 70B + 3B
    system_prompt_calls = n_prompts_a * 3 * 2  # 3 seeds, with/without
    temporal_calls = 3
    adversarial_calls = 5

    total_venice = group_a_calls + group_b_calls + system_prompt_calls + temporal_calls + adversarial_calls
    total_together = n_prompts_a * n_seeds + n_prompts_b * n_seeds + adversarial_calls

    return {
        "venice_calls": total_venice,
        "together_calls": total_together,
        "estimated_venice_cost": f"${total_venice * cost_per_call:.2f}",
        "estimated_together_cost": f"${total_together * cost_per_call:.2f}",
    }


def main():
    parser = argparse.ArgumentParser(description="VerifyVenice data collection")
    parser.add_argument("--dry-run", action="store_true", help="Single call per provider to test connectivity")
    parser.add_argument("--group", choices=["a", "b", "all"], default="all", help="Which test group to run")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip local calibration")
    args = parser.parse_args()

    config = VerifyConfig.load()
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Cost estimate
    costs = estimate_cost(config)
    print("Estimated costs:")
    for k, v in costs.items():
        print(f"  {k}: {v}")
    print()

    exporter = DataExporter()
    venice = VeniceClient(config)
    reference = ReferenceClient(config)

    # Check Ollama availability
    local = OllamaClient(config)
    ollama_available = local.is_available()
    if not ollama_available:
        print("WARNING: Ollama not available. 3B local comparison will be skipped.")
        local = None

    # Calibration
    if not args.skip_calibration and local is not None and not args.dry_run:
        print("\n=== Phase 0: Calibration ===")
        calibrator = QuantizationCalibrator(config)
        cal_results = calibrator.run_calibration(local)
        exporter.save_json(cal_results, config.data_dir / "processed" / "calibration.json")
        print(f"Calibration complete. Intra-model KL mean: {cal_results['intra_model_kl']['mean']:.6f}")

    # Group A
    if args.group in ("a", "all"):
        print("\n=== Group A: Output Integrity ===")
        collector_a = OutputIntegrityCollector(config, venice, reference, local)
        results_a = collector_a.collect(dry_run=args.dry_run)
        path_a = collector_a.save_all_results("group_a_results.json")
        print(f"Group A: {len(results_a)} results saved to {path_a}")

        if not args.dry_run:
            print("\n=== System Prompt Differential ===")
            sys_prompt_results = collector_a.collect_system_prompt_differential()
            exporter.save_json(
                sys_prompt_results,
                config.data_dir / "processed" / "system_prompt_diff.json",
            )
            print(f"System prompt differential: {len(sys_prompt_results)} results saved")

    # Group B
    if args.group in ("b", "all"):
        print("\n=== Group B: Model Authenticity ===")
        collector_b = ModelAuthenticityCollector(config, venice, reference, local)
        results_b = collector_b.collect(dry_run=args.dry_run)
        path_b = collector_b.save_all_results("group_b_results.json")
        print(f"Group B: {len(results_b)} results saved to {path_b}")

        if not args.dry_run:
            print("\n=== Temporal Consistency ===")
            temporal_results = collector_b.collect_temporal_consistency()
            exporter.save_json(
                temporal_results,
                config.data_dir / "processed" / "temporal_consistency.json",
            )

            print("\n=== Adversarial Probes ===")
            adversarial_results = collector_b.collect_adversarial()
            exporter.save_json(
                adversarial_results,
                config.data_dir / "processed" / "adversarial_results.json",
            )

    print("\nCollection complete.")


if __name__ == "__main__":
    main()
