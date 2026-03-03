"""Generate the final verification report.

Usage:
    python scripts/run_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import VerifyConfig
from src.report.generator import ReportGenerator


def main():
    config = VerifyConfig.load()

    analysis_path = config.data_dir / "processed" / "analysis_results.json"
    if not analysis_path.exists():
        print(f"Analysis results not found at {analysis_path}")
        print("Run `python scripts/run_analysis.py` first.")
        sys.exit(1)

    with open(analysis_path) as f:
        analysis_results = json.load(f)

    # Check for proof results
    proof_results = {}
    proof_path = config.data_dir / "proofs" / "proof_summary.json"
    if proof_path.exists():
        with open(proof_path) as f:
            proof_results = json.load(f)

    generator = ReportGenerator()
    report = generator.generate(analysis_results, proof_results)
    output_path = generator.save(report, config.reports_dir)

    print(f"Report generated: {output_path}")
    print(f"Report length: {len(report)} characters")


if __name__ == "__main__":
    main()
