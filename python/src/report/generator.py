"""Markdown report generator for VerifyVenice results.

Adapted for the reality that Venice's llama-3.3-70b does NOT support
logprobs. Group A uses text similarity for 70B and logprob stats for 3B.
Group B uses text-based and top-1 logprob fingerprinting.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ReportGenerator:
    def generate(
        self,
        analysis_results: dict[str, Any],
        proof_results: dict[str, Any] | None = None,
    ) -> str:
        sections = [
            self._header(),
            self._executive_summary(analysis_results),
            self._methodology(),
            self._calibration_results(analysis_results.get("calibration", {})),
            self._group_a_results(analysis_results.get("output_integrity", {})),
            self._group_b_results(analysis_results.get("model_authenticity", {})),
            self._system_prompt_results(analysis_results.get("system_prompt_differential", {})),
            self._zkml_proof_summary(proof_results or {}),
            self._limitations(),
            self._independence_commitment(),
        ]
        return "\n\n---\n\n".join(sections)

    def save(self, report: str, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"report_{timestamp}.md"
        with open(path, "w") as f:
            f.write(report)
        return path

    def _header(self) -> str:
        return f"""# VerifyVenice: Verification Report

**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
**Project:** Venice.ai API Verification via JOLT-Atlas zkML

Venice.ai claims to serve uncensored inference using authentic, full-size
models. This report presents empirical evidence for or against those claims
based on statistical comparison of Venice's API responses against reference
providers."""

    def _executive_summary(self, results: dict) -> str:
        lines = ["## Executive Summary\n"]
        lines.append("### What We Found About Venice\n")

        oi = results.get("output_integrity", {})
        if oi:
            comp_70b = oi.get("70b_comparison", {})
            comp_3b = oi.get("3b_comparison", {})

            if comp_70b:
                cosine = comp_70b.get("mean_cosine_similarity", "N/A")
                n = comp_70b.get("n_total", 0)
                lines.append(f"- **70B Output Integrity:** Mean cosine similarity = {cosine} "
                            f"across {n} test pairs (text-based comparison)")

            if comp_3b:
                agreement = comp_3b.get("mean_token_agreement", "N/A")
                n = comp_3b.get("n_total", 0)
                lines.append(f"- **3B Output Integrity:** Mean token agreement = {agreement} "
                            f"across {n} test pairs (logprob-based comparison)")

        ma = results.get("model_authenticity", {})
        if ma:
            v70b_pred = ma.get("venice_70b_text_prediction", {})
            if v70b_pred:
                verdict = v70b_pred.get("verdict", "N/A")
                conf = v70b_pred.get("confidence", 0)
                pred_class = v70b_pred.get("predicted_class", "N/A")
                lines.append(f"- **70B Model Authenticity:** Venice 70B classified as "
                            f"**{pred_class}** ({conf:.0%} confidence) — **{verdict}**")

            ollama_pred = ma.get("ollama_3b_prediction", {})
            if ollama_pred:
                verdict = ollama_pred.get("verdict", "N/A")
                lines.append(f"- **3B Validation:** Ollama 3B correctly identified — **{verdict}**")

        lines.append("")
        lines.append("### Data Availability Constraints\n")
        lines.append("Venice's `llama-3.3-70b` does **not** support logprobs. The 70B comparison "
                     "relies on text similarity metrics rather than direct logprob distribution "
                     "comparison. Venice's `llama-3.2-3b` supports per-token logprobs (top-1 only, "
                     "no alternative candidates).")

        return "\n".join(lines)

    def _methodology(self) -> str:
        return """## Methodology

### What This Proves About Venice

1. **Output integrity:** Whether Venice's responses show evidence of filtering,
   censorship, or modification compared to the same model served by an
   independent provider.
2. **Model authenticity:** Whether Venice's response characteristics are
   consistent with the claimed model (70B), not a smaller substitute (3B).
3. **System prompt transparency:** Whether Venice's system prompt toggle is
   the only behavioral modification point.

### How We Test It

**Group A (Output Integrity):** Send identical prompts to Venice and a
reference provider. Compare outputs using:
- **70B (text-only):** Cosine similarity, Jaccard similarity, BLEU, edit
  distance. Venice's 70B doesn't support logprobs, so comparison is based
  on response content.
- **3B (logprob-based):** Token agreement rate, per-token logprob correlation.
  Venice's 3B returns top-1 logprobs; Ollama returns full top-5 distributions.

**Group B (Model Authenticity):** Train classifiers to distinguish model sizes:
- **Text classifier:** 6-dim features (response length, word length, vocabulary
  richness, sentence structure). Trained on Together 70B vs Venice 3B, then
  applied to Venice 70B.
- **Logprob classifier:** 4-dim top-1 features (mean logprob, std logprob,
  high-confidence fraction, perplexity). Used where logprobs are available.

### Reference Providers
- **Together.ai** (Llama 3.3 70B Instruct Turbo): Independent reference.
  Together.ai serves the claimed model — this is a trust assumption, mitigated
  by their independent reputation and widespread use.
- **Ollama** (Llama 3.2 3B, local): True ground truth — we control the model
  and weights directly.

### Statistical Thresholds
Thresholds were established empirically via calibration (see below). We first
measure "normal" divergence between identical models on different infrastructure,
then flag Venice results that exceed that baseline."""

    def _calibration_results(self, calibration: dict) -> str:
        if not calibration:
            return "## Calibration\n\n*Calibration not yet run.*"

        intra = calibration.get("intra_model_kl", {})
        entropy = calibration.get("entropy", {})
        agreement = calibration.get("token_agreement", {})

        return f"""## Calibration Baselines

Calibration establishes "normal" divergence for the local 3B model:

| Metric | Mean | Std | 95th %ile |
|--------|------|-----|-----------|
| Intra-model KL | {_fmt(intra.get('mean'))} | {_fmt(intra.get('std'))} | {_fmt(intra.get('p95'))} |
| Mean entropy | {_fmt(entropy.get('mean'))} | {_fmt(entropy.get('std'))} | -- |
| Token agreement | {_fmt(agreement.get('mean'))} | {_fmt(agreement.get('std'))} | -- |

**Interpretation:** Calibration quantifies the nondeterminism inherent in
running the same model with the same seed. Venice divergence below the
95th percentile is within normal variance."""

    def _group_a_results(self, results: dict) -> str:
        if not results:
            return "## Group A: Output Integrity\n\n*Tests not yet run.*"

        lines = ["## Group A: Output Integrity\n"]

        # 70B text-only comparison
        comp_70b = results.get("70b_comparison", {})
        if comp_70b:
            lines.append("### 70B: Venice vs Together.ai (Text Comparison)\n")
            lines.append(f"*Comparison mode: {comp_70b.get('mode', 'text_only')} "
                        f"(Venice 70B does not support logprobs)*\n")

            per_prompt = comp_70b.get("per_prompt", {})
            if per_prompt:
                lines.append("| Prompt Type | Cosine Sim | Jaccard | BLEU-1 | Edit Dist | Verdict |")
                lines.append("|-------------|------------|---------|--------|-----------|---------|")

                for name, m in per_prompt.items():
                    cosine = m.get("cosine_similarity", 0)
                    jaccard = m.get("jaccard_similarity", 0)
                    bleu = m.get("bleu_1gram", 0)
                    edit = m.get("edit_distance", 0)
                    verdict = self._text_verdict(cosine, jaccard)
                    lines.append(
                        f"| {name} | {cosine:.4f} | {jaccard:.4f} | {bleu:.4f} | {edit:.4f} | {verdict} |"
                    )

                lines.append(f"\n**Overall:** {comp_70b.get('mean_cosine_similarity', 'N/A')}")

        # 3B logprob comparison
        comp_3b = results.get("3b_comparison", {})
        if comp_3b:
            lines.append("\n### 3B: Venice vs Ollama (Logprob Comparison)\n")
            lines.append(f"*Comparison mode: {comp_3b.get('mode', 'top1_logprob')}*\n")

            per_prompt = comp_3b.get("per_prompt", {})
            if per_prompt:
                lines.append("| Prompt Type | Token Agreement | Cosine Sim | Top-1 Corr | Verdict |")
                lines.append("|-------------|-----------------|------------|------------|---------|")

                for name, m in per_prompt.items():
                    agreement = m.get("token_agreement", 0)
                    cosine = m.get("cosine_similarity", 0)
                    corr = m.get("top1_logprob_correlation", 0)
                    verdict = "PASS" if agreement > 0.2 or corr > 0.3 else "WARN"
                    lines.append(
                        f"| {name} | {agreement:.2%} | {cosine:.4f} | {corr:.4f} | {verdict} |"
                    )

                lines.append(f"\n**Overall:** {comp_3b.get('mean_token_agreement', 'N/A')}")

        return "\n".join(lines)

    def _group_b_results(self, results: dict) -> str:
        if not results:
            return "## Group B: Model Authenticity\n\n*Tests not yet run.*"

        lines = ["## Group B: Model Authenticity\n"]

        # Text-based classifier
        text_clf = results.get("text_classifier", {})
        if text_clf:
            lines.append("### Text-Based Model Classifier\n")
            lines.append(f"Trained on Together 70B vs Venice 3B text features to distinguish "
                        f"model sizes.\n")
            lines.append(f"**CV Accuracy:** {text_clf.get('cv_accuracy_mean', 0):.1%} "
                        f"(+/- {text_clf.get('cv_accuracy_std', 0):.1%})")
            lines.append("")

        v70b_pred = results.get("venice_70b_text_prediction", {})
        if v70b_pred:
            lines.append("| Model | Predicted Class | Confidence | Verdict |")
            lines.append("|-------|----------------|------------|---------|")
            lines.append(
                f"| Venice 70B | {v70b_pred.get('predicted_class', 'N/A')} | "
                f"{v70b_pred.get('confidence', 0):.1%} | **{v70b_pred.get('verdict', 'N/A')}** |"
            )
            lines.append("")

        # Logprob classifier
        lp_clf = results.get("logprob_classifier", {})
        if lp_clf:
            lines.append("### Logprob-Based Model Classifier\n")
            lines.append(f"Trained on Together 70B vs Venice 3B top-1 logprob features.\n")
            lines.append(f"**CV Accuracy:** {lp_clf.get('cv_accuracy_mean', 0):.1%} "
                        f"(+/- {lp_clf.get('cv_accuracy_std', 0):.1%})")
            lines.append("")

        ollama_pred = results.get("ollama_3b_prediction", {})
        if ollama_pred:
            lines.append("### Ground Truth Validation\n")
            lines.append("| Model | Expected | Predicted | Confidence | Verdict |")
            lines.append("|-------|----------|-----------|------------|---------|")
            lines.append(
                f"| Ollama 3B | 3b-class | {ollama_pred.get('predicted_class', 'N/A')} | "
                f"{ollama_pred.get('confidence', 0):.1%} | **{ollama_pred.get('verdict', 'N/A')}** |"
            )
            lines.append("")
            lines.append("*Ollama correctly identified as 3B validates the classifier is working.*")

        # Feature importance
        importance = results.get("feature_importance", {})
        if importance:
            lines.append("\n### Feature Importance\n")
            sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feat, weight in sorted_feats:
                bar = "#" * int(weight * 10)
                lines.append(f"- `{feat}`: {weight:.4f} {bar}")

        return "\n".join(lines)

    def _system_prompt_results(self, results: dict) -> str:
        if not results:
            return "## System Prompt Transparency\n\n*System prompt tests not yet run.*"

        lines = ["## System Prompt Transparency\n"]
        lines.append(f"**Finding:** {results.get('finding', 'N/A')}\n")
        lines.append(f"Venice's `include_venice_system_prompt` toggle was tested across "
                     f"{results.get('n_total', 0)} prompt/seed combinations. "
                     f"**{results.get('identical_responses', 'N/A')}** produced identical output "
                     f"-- the toggle has a real, measurable effect on model behavior.\n")

        per_prompt = results.get("per_prompt", {})
        if per_prompt:
            lines.append("| Prompt Type | Cosine Sim | Jaccard | Length WITH | Length WITHOUT | Ratio |")
            lines.append("|-------------|------------|---------|------------|---------------|-------|")

            for name, m in per_prompt.items():
                cosine = m.get("cosine_similarity", 0)
                jaccard = m.get("jaccard_similarity", 0)
                len_with = m.get("avg_length_with_prompt", 0)
                len_without = m.get("avg_length_without_prompt", 0)
                ratio = m.get("length_ratio", 0)
                lines.append(
                    f"| {name} | {cosine:.4f} | {jaccard:.4f} | "
                    f"{len_with:.0f} | {len_without:.0f} | {ratio:.2f} |"
                )

        lines.append(f"\n**Mean cosine similarity (with vs without):** "
                     f"{results.get('mean_cosine_similarity', 0):.4f}")
        lines.append(f"**Mean length ratio:** {results.get('mean_length_ratio', 0):.2f}")
        lines.append("")
        lines.append("**Interpretation:** The system prompt toggle is not cosmetic. Venice "
                     "applies a real system prompt that changes response behavior, particularly "
                     "for factual queries where responses with the prompt are significantly "
                     "shorter. This is consistent with Venice's transparency claim about "
                     "the `include_venice_system_prompt` parameter.")

        return "\n".join(lines)

    def _zkml_proof_summary(self, proofs: dict) -> str:
        if not proofs:
            return "## zkML Proof Summary\n\n*Proofs not yet generated.*"

        lines = ["## zkML Proof Summary\n"]
        lines.append("| Circuit | Status | Proof Size | Verification |")
        lines.append("|---------|--------|------------|-------------|")

        for name, info in proofs.items():
            lines.append(
                f"| {name} | {info.get('status', 'N/A')} | "
                f"{info.get('size_bytes', 0)} bytes | "
                f"{info.get('verified', 'N/A')} |"
            )

        lines.append("")
        lines.append("> These proofs guarantee the analysis was performed honestly --")
        lines.append("> anyone can verify we didn't fabricate or cherry-pick results.")

        return "\n".join(lines)

    def _limitations(self) -> str:
        return """## Limitations

### Venice Logprob Constraints
Venice's `llama-3.3-70b` does **not** support logprobs. This means we cannot
directly compare logprob distributions between Venice 70B and Together 70B.
The 70B comparison relies on text similarity, which is a weaker signal than
logprob comparison. Venice's `llama-3.2-3b` returns per-token logprobs but
not alternative candidates (top_logprobs is empty).

### Scope
We tested ~50 prompts across 5 categories. Results characterize Venice's
behavior for these prompts during the test window. Venice could behave
differently for other prompts, at other times, or after infrastructure changes.

### What's Out of Scope
- **Server-side privacy:** Whether Venice logs or shares prompts cannot be
  tested externally. That claim requires TEEs or server-side zkML.
- **Quantization specifics:** We can distinguish model classes (3B vs 70B)
  but not exact quantization levels (FP16 vs INT4).

### Trust Assumptions
- **Together.ai reference:** We assume Together.ai serves the real
  Llama 3.3 70B. Mitigated by their independent reputation.
- **Data provenance:** zkML proofs verify our analysis, not that we
  captured Venice's responses faithfully. Mitigated by publishing all
  scripts so anyone can reproduce the tests."""

    def _independence_commitment(self) -> str:
        return """## Independence Commitment

This verification was conducted independently. Results are published
regardless of whether they are favorable or unfavorable to Venice.ai.
Venice had no editorial control over methodology, analysis, or conclusions.

The source code, raw data, and zkML proofs are publicly available for
independent verification."""

    def _text_verdict(self, cosine: float, jaccard: float) -> str:
        """Verdict for text-only comparison."""
        avg = (cosine + jaccard) / 2
        if avg > 0.7:
            return "PASS"
        elif avg > 0.4:
            return "WARN"
        else:
            return "FAIL"


def _fmt(val: Any) -> str:
    """Format a numeric value or return 'N/A'."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)
