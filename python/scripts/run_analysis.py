"""Main analysis script. Runs after data collection.

Handles three comparison modes:
- Full logprobs (Together, Ollama): KL divergence, entropy, token agreement
- Top-1 logprobs (Venice 3B): Per-token logprob stats, token agreement
- Text-only (Venice 70B): Cosine similarity, BLEU, edit distance

Usage:
    python scripts/run_analysis.py [--data-dir DATA_DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.analysis.fingerprint import ModelFingerprinter
from src.analysis.similarity import TextSimilarity
from src.analysis.statistical import StatisticalAnalyzer
from src.config import VerifyConfig
from src.exporters.data_export import DataExporter


def load_json(path: Path) -> list | dict:
    with open(path) as f:
        return json.load(f)


def analyze_output_integrity(
    results: list[dict],
    analyzer: StatisticalAnalyzer,
    similarity: TextSimilarity,
    config: VerifyConfig,
) -> dict:
    """Analyze Group A results.

    Handles both text-only (70B) and logprob-based (3B) comparisons.
    """
    per_prompt_70b: dict[str, dict] = {}
    per_prompt_3b: dict[str, dict] = {}
    all_cosine: list[float] = []
    all_kl_3b: list[float] = []
    all_agreement_3b: list[float] = []

    for result in results:
        prompt_type = result["prompt_type"]
        venice = result.get("venice", {})
        reference = result.get("reference", {})

        # 70B comparison: always text-based (Venice 70B has no logprobs)
        venice_text = venice.get("response_text", "")
        ref_text = reference.get("response_text", "")

        if venice_text and ref_text:
            cosine = similarity.cosine_similarity(venice_text, ref_text)
            jaccard = similarity.jaccard_token_similarity(venice_text, ref_text)
            edit_dist = similarity.edit_distance_normalized(venice_text, ref_text)
            bleu = similarity.bleu_1gram(ref_text, venice_text)

            # Token agreement from logprobs (works even when only one side has logprobs)
            v_lp = venice.get("logprobs", [])
            r_lp = reference.get("logprobs", [])
            token_agreement = analyzer.token_agreement_rate(v_lp, r_lp) if v_lp and r_lp else None

            if prompt_type not in per_prompt_70b:
                per_prompt_70b[prompt_type] = {
                    "cosine_values": [], "jaccard_values": [],
                    "edit_distance_values": [], "bleu_values": [],
                    "token_agreement_values": [],
                }

            per_prompt_70b[prompt_type]["cosine_values"].append(cosine)
            per_prompt_70b[prompt_type]["jaccard_values"].append(jaccard)
            per_prompt_70b[prompt_type]["edit_distance_values"].append(edit_dist)
            per_prompt_70b[prompt_type]["bleu_values"].append(bleu)
            if token_agreement is not None:
                per_prompt_70b[prompt_type]["token_agreement_values"].append(token_agreement)
            all_cosine.append(cosine)

        # 3B comparison (if available): logprob-based + text
        venice_3b = result.get("venice_3b", {})
        local_3b = result.get("local_3b", {})

        if venice_3b and local_3b:
            v3b_lp = venice_3b.get("logprobs", [])
            l3b_lp = local_3b.get("logprobs", [])

            agreement_3b = analyzer.token_agreement_rate(v3b_lp, l3b_lp) if v3b_lp and l3b_lp else 0.0
            cosine_3b = similarity.cosine_similarity(
                venice_3b.get("response_text", ""),
                local_3b.get("response_text", ""),
            )

            # KL divergence only if both have top_logprobs
            has_top_lp = (
                v3b_lp and l3b_lp
                and any(t.get("top_logprobs") for t in v3b_lp)
                and any(t.get("top_logprobs") for t in l3b_lp)
            )
            kl_3b = analyzer.kl_divergence(v3b_lp, l3b_lp) if has_top_lp else None

            # Top-1 logprob correlation
            v3b_top1 = [t.get("logprob", 0) for t in v3b_lp] if v3b_lp else []
            l3b_top1 = [t.get("logprob", 0) for t in l3b_lp] if l3b_lp else []
            min_len = min(len(v3b_top1), len(l3b_top1))
            top1_corr = float(np.corrcoef(v3b_top1[:min_len], l3b_top1[:min_len])[0, 1]) if min_len >= 2 else 0.0

            if prompt_type not in per_prompt_3b:
                per_prompt_3b[prompt_type] = {
                    "agreement_values": [], "cosine_values": [],
                    "kl_values": [], "top1_corr_values": [],
                }

            per_prompt_3b[prompt_type]["agreement_values"].append(agreement_3b)
            per_prompt_3b[prompt_type]["cosine_values"].append(cosine_3b)
            if kl_3b is not None:
                per_prompt_3b[prompt_type]["kl_values"].append(kl_3b)
                all_kl_3b.append(kl_3b)
            per_prompt_3b[prompt_type]["top1_corr_values"].append(top1_corr)
            all_agreement_3b.append(agreement_3b)

    # Summarize 70B (text-only)
    summary_70b: dict[str, dict] = {}
    for prompt_type, values in per_prompt_70b.items():
        summary_70b[prompt_type] = {
            "cosine_similarity": float(np.mean(values["cosine_values"])),
            "jaccard_similarity": float(np.mean(values["jaccard_values"])),
            "edit_distance": float(np.mean(values["edit_distance_values"])),
            "bleu_1gram": float(np.mean(values["bleu_values"])),
            "n_samples": len(values["cosine_values"]),
        }
        if values["token_agreement_values"]:
            summary_70b[prompt_type]["token_agreement"] = float(np.mean(values["token_agreement_values"]))

    # Summarize 3B
    summary_3b: dict[str, dict] = {}
    for prompt_type, values in per_prompt_3b.items():
        summary_3b[prompt_type] = {
            "token_agreement": float(np.mean(values["agreement_values"])),
            "cosine_similarity": float(np.mean(values["cosine_values"])),
            "top1_logprob_correlation": float(np.mean(values["top1_corr_values"])),
            "n_samples": len(values["agreement_values"]),
        }
        if values["kl_values"]:
            summary_3b[prompt_type]["kl_divergence"] = float(np.mean(values["kl_values"]))

    # Overall CIs
    cos_ci = analyzer.bootstrap_confidence_interval(all_cosine) if all_cosine else (0, 0, 0)

    output: dict[str, Any] = {
        "70b_comparison": {
            "mode": "text_only",
            "per_prompt": summary_70b,
            "mean_cosine_similarity": f"{cos_ci[0]:.4f} [{cos_ci[1]:.4f}, {cos_ci[2]:.4f}]",
            "n_total": len(all_cosine),
        },
    }

    if summary_3b:
        agr_ci = analyzer.bootstrap_confidence_interval(all_agreement_3b) if all_agreement_3b else (0, 0, 0)
        output["3b_comparison"] = {
            "mode": "top1_logprob" if not all_kl_3b else "full_logprob",
            "per_prompt": summary_3b,
            "mean_token_agreement": f"{agr_ci[0]:.4f} [{agr_ci[1]:.4f}, {agr_ci[2]:.4f}]",
            "n_total": len(all_agreement_3b),
        }
        if all_kl_3b:
            kl_ci = analyzer.bootstrap_confidence_interval(all_kl_3b)
            output["3b_comparison"]["mean_kl_divergence"] = f"{kl_ci[0]:.4f} [{kl_ci[1]:.4f}, {kl_ci[2]:.4f}]"

    return output


def analyze_model_authenticity(
    results: list[dict],
    fingerprinter: ModelFingerprinter,
) -> dict:
    """Analyze Group B results.

    Uses text-based features for Venice 70B (no logprobs),
    logprob features for Together 70B and Venice 3B.
    """
    venice_70b_responses = [r["venice_70b"] for r in results if "venice_70b" in r]
    venice_3b_responses = [r["venice_3b"] for r in results if "venice_3b" in r]
    together_70b_responses = [r["together_70b"] for r in results if "together_70b" in r]
    ollama_3b_responses = [r["ollama_3b"] for r in results if "ollama_3b" in r]

    analysis: dict[str, Any] = {}

    # --- Logprob-based fingerprinting (Together 70B vs Venice/Ollama 3B) ---
    # Train on Together 70B (full logprobs) vs Venice 3B (top-1 logprobs).
    # Use top-1 features since Venice 3B only has those.
    t70b_top1 = fingerprinter.build_top1_feature_matrix(together_70b_responses)
    v3b_top1 = fingerprinter.build_top1_feature_matrix(venice_3b_responses)

    if len(t70b_top1) >= 2 and len(v3b_top1) >= 2:
        train_features = np.vstack([t70b_top1, v3b_top1])
        train_labels = np.array([0] * len(t70b_top1) + [1] * len(v3b_top1))

        if len(train_features) >= 4:
            training_metrics = fingerprinter.train_classifier(
                train_features, train_labels,
                feature_names=fingerprinter.TOP1_FEATURE_NAMES,
            )
            analysis["logprob_classifier"] = training_metrics

            # Predict on Ollama 3B (should classify as 3b-class)
            if ollama_3b_responses:
                ollama_top1 = fingerprinter.build_top1_feature_matrix(ollama_3b_responses)
                if len(ollama_top1) > 0:
                    pred_labels, pred_probs = fingerprinter.predict(ollama_top1)
                    analysis["ollama_3b_prediction"] = {
                        "expected_class": "3b-class",
                        "predicted_class": "70b-class" if np.median(pred_labels) == 0 else "3b-class",
                        "confidence": float(np.mean(np.max(pred_probs, axis=1))),
                        "verdict": "PASS" if np.median(pred_labels) == 1 else "FAIL",
                    }

    # --- Text-based fingerprinting (Venice 70B vs Together 70B vs Venice 3B) ---
    v70b_text = fingerprinter.build_text_feature_matrix(venice_70b_responses)
    t70b_text = fingerprinter.build_text_feature_matrix(together_70b_responses)
    v3b_text = fingerprinter.build_text_feature_matrix(venice_3b_responses)

    text_fp = ModelFingerprinter()  # separate instance for text classifier
    if len(t70b_text) >= 2 and len(v3b_text) >= 2:
        text_train = np.vstack([t70b_text, v3b_text])
        text_labels = np.array([0] * len(t70b_text) + [1] * len(v3b_text))

        if len(text_train) >= 4:
            text_metrics = text_fp.train_classifier(
                text_train, text_labels,
                feature_names=fingerprinter.TEXT_FEATURE_NAMES,
            )
            analysis["text_classifier"] = text_metrics

            # Classify Venice 70B — should look like 70b-class (0)
            if len(v70b_text) > 0:
                pred_labels, pred_probs = text_fp.predict(v70b_text)
                class_names = {0: "70b-class", 1: "3b-class"}
                analysis["venice_70b_text_prediction"] = {
                    "predicted_class": class_names.get(int(np.median(pred_labels)), "unknown"),
                    "confidence": float(np.mean(np.max(pred_probs, axis=1))),
                    "class_distribution": {
                        class_names.get(c, str(c)): float(np.mean(pred_labels == c))
                        for c in np.unique(pred_labels)
                    },
                    "verdict": "PASS" if np.median(pred_labels) == 0 else "FAIL",
                }

    # Feature summaries
    analysis["feature_summary"] = {}
    if len(v70b_text) > 0:
        analysis["feature_summary"]["venice_70b_text"] = _feature_summary(v70b_text)
    if len(t70b_text) > 0:
        analysis["feature_summary"]["together_70b_text"] = _feature_summary(t70b_text)
    if len(v3b_text) > 0:
        analysis["feature_summary"]["venice_3b_text"] = _feature_summary(v3b_text)
    if len(t70b_top1) > 0:
        analysis["feature_summary"]["together_70b_logprob"] = _feature_summary(t70b_top1)
    if len(v3b_top1) > 0:
        analysis["feature_summary"]["venice_3b_logprob"] = _feature_summary(v3b_top1)

    analysis["feature_importance"] = fingerprinter.get_feature_importance()

    return analysis


def analyze_system_prompt_diff(
    results: list[dict],
    similarity: TextSimilarity,
) -> dict:
    """Analyze system prompt on vs off differential."""
    per_prompt: dict[str, dict] = {}
    all_cosine: list[float] = []
    all_len_ratios: list[float] = []

    for result in results:
        prompt_type = result["prompt_type"]
        with_text = result["with_venice_prompt"].get("response_text", "")
        without_text = result["without_venice_prompt"].get("response_text", "")

        if not with_text or not without_text:
            continue

        cosine = similarity.cosine_similarity(with_text, without_text)
        jaccard = similarity.jaccard_token_similarity(with_text, without_text)
        edit = similarity.edit_distance_normalized(with_text, without_text)
        identical = with_text == without_text
        len_ratio = len(with_text) / len(without_text) if without_text else 0.0

        if prompt_type not in per_prompt:
            per_prompt[prompt_type] = {
                "cosine_values": [], "jaccard_values": [], "edit_values": [],
                "with_lengths": [], "without_lengths": [],
                "identical_count": 0, "total": 0,
            }

        per_prompt[prompt_type]["cosine_values"].append(cosine)
        per_prompt[prompt_type]["jaccard_values"].append(jaccard)
        per_prompt[prompt_type]["edit_values"].append(edit)
        per_prompt[prompt_type]["with_lengths"].append(len(with_text))
        per_prompt[prompt_type]["without_lengths"].append(len(without_text))
        per_prompt[prompt_type]["identical_count"] += int(identical)
        per_prompt[prompt_type]["total"] += 1
        all_cosine.append(cosine)
        all_len_ratios.append(len_ratio)

    summary: dict[str, dict] = {}
    n_identical = 0
    n_total = 0
    for prompt_type, v in per_prompt.items():
        avg_with = float(np.mean(v["with_lengths"]))
        avg_without = float(np.mean(v["without_lengths"]))
        summary[prompt_type] = {
            "cosine_similarity": float(np.mean(v["cosine_values"])),
            "jaccard_similarity": float(np.mean(v["jaccard_values"])),
            "edit_distance": float(np.mean(v["edit_values"])),
            "avg_length_with_prompt": avg_with,
            "avg_length_without_prompt": avg_without,
            "length_ratio": avg_with / avg_without if avg_without > 0 else 0.0,
            "identical_responses": f"{v['identical_count']}/{v['total']}",
        }
        n_identical += v["identical_count"]
        n_total += v["total"]

    return {
        "per_prompt": summary,
        "mean_cosine_similarity": float(np.mean(all_cosine)) if all_cosine else 0.0,
        "mean_length_ratio": float(np.mean(all_len_ratios)) if all_len_ratios else 0.0,
        "identical_responses": f"{n_identical}/{n_total}",
        "n_total": n_total,
        "finding": "Venice system prompt toggle produces measurably different behavior"
        if n_identical == 0 else "Some responses identical with/without system prompt",
    }


def _feature_summary(features: np.ndarray) -> dict:
    return {
        "mean": features.mean(axis=0).tolist(),
        "std": features.std(axis=0).tolist(),
        "n": len(features),
    }


def main():
    parser = argparse.ArgumentParser(description="VerifyVenice analysis")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    config = VerifyConfig.load()
    data_dir = args.data_dir or config.data_dir / "processed"
    exporter = DataExporter()

    analyzer = StatisticalAnalyzer()
    similarity = TextSimilarity()
    fingerprinter = ModelFingerprinter()

    all_results: dict[str, dict] = {}

    # Load calibration
    cal_path = data_dir / "calibration.json"
    if cal_path.exists():
        all_results["calibration"] = load_json(cal_path)
        print(f"Loaded calibration from {cal_path}")

    # Group A
    group_a_path = data_dir / "group_a_results.json"
    if group_a_path.exists():
        group_a_data = load_json(group_a_path)
        all_results["output_integrity"] = analyze_output_integrity(
            group_a_data, analyzer, similarity, config
        )
        n70 = all_results["output_integrity"]["70b_comparison"]["n_total"]
        n3 = all_results["output_integrity"].get("3b_comparison", {}).get("n_total", 0)
        print(f"Group A: Analyzed {n70} 70B pairs (text-only), {n3} 3B pairs")

    # Group B
    group_b_path = data_dir / "group_b_results.json"
    if group_b_path.exists():
        group_b_data = load_json(group_b_path)
        all_results["model_authenticity"] = analyze_model_authenticity(
            group_b_data, fingerprinter
        )
        lp_clf = all_results["model_authenticity"].get("logprob_classifier", {})
        txt_clf = all_results["model_authenticity"].get("text_classifier", {})
        print(f"Group B: logprob classifier acc={lp_clf.get('cv_accuracy_mean', 'N/A')}, "
              f"text classifier acc={txt_clf.get('cv_accuracy_mean', 'N/A')}")

        # Save feature matrix (text features for all)
        venice_70b = [r["venice_70b"] for r in group_b_data if "venice_70b" in r]
        venice_3b = [r["venice_3b"] for r in group_b_data if "venice_3b" in r]
        together_70b = [r["together_70b"] for r in group_b_data if "together_70b" in r]

        all_responses = venice_70b + venice_3b + together_70b
        all_features = fingerprinter.build_text_feature_matrix(all_responses)
        all_labels = np.array(
            [0] * len(venice_70b) + [1] * len(venice_3b) + [0] * len(together_70b)
        )
        exporter.save_feature_matrix(
            all_features, all_labels, fingerprinter.TEXT_FEATURE_NAMES,
            data_dir / "feature_matrix.npz",
        )

    # System prompt differential
    sysprompt_path = data_dir / "system_prompt_diff.json"
    if sysprompt_path.exists():
        sysprompt_data = load_json(sysprompt_path)
        all_results["system_prompt_differential"] = analyze_system_prompt_diff(
            sysprompt_data, similarity
        )
        n_sp = all_results["system_prompt_differential"]["n_total"]
        print(f"System prompt: Analyzed {n_sp} differential pairs")

    # Save analysis results
    output_path = exporter.save_json(all_results, data_dir / "analysis_results.json")
    print(f"\nAnalysis results saved to {output_path}")


if __name__ == "__main__":
    main()
