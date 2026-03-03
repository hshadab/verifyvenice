# Limitations and Trust Assumptions

## What This Verification Proves About Venice

1. **Output integrity:** For the tested prompts, Venice's responses
   are or aren't statistically consistent with the same model served
   by an independent reference provider. If Venice filters, censors,
   or modifies model output, the divergence is measurably higher than
   the calibration baseline.

2. **Model authenticity:** Venice's logprob distributions are or aren't
   consistent with the claimed model family. If Venice substitutes a
   3B model when claiming 70B, the fingerprint classifier catches it.

3. **System prompt behavior:** Venice's `include_venice_system_prompt`
   toggle does or doesn't behave as documented — and we measure the
   exact statistical impact of their system prompt on output.

4. **Analysis honesty:** The zkML proofs guarantee our comparisons
   were performed correctly. Nobody has to trust our numbers — they
   can verify the proofs.

## Scope Boundaries

### Sample Size
We tested ~50 prompts across 5 categories. This characterizes Venice's
behavior for these specific inputs during the test window. We are not
claiming Venice "never filters" or "always serves 70B" — those are
universal claims that would require continuous monitoring, not a
point-in-time test.

### Server-Side Privacy
Whether Venice logs, stores, or shares prompts cannot be tested from
the outside. This is Venice's core privacy claim and it remains
unverifiable without:
- TEE attestation (Venice already has `supportsTeeAttestation` in
  their model spec — not yet active)
- Server-side zkML integration
- Independent code audit

### Temporal Validity
Results reflect Venice's behavior during our test window. Infrastructure
or policy changes after testing could invalidate findings.

## Trust Assumptions

### Together.ai as Reference
We use Together.ai's Llama 3.3 70B as a reference. This assumes
Together.ai faithfully serves the claimed model. Mitigations:
- Together.ai has independent reputation and widespread research use
- Ollama provides true local ground truth for the 3B model class
- Cross-referencing two independent providers is stronger than one

### Quantization Variance
Production models are almost always quantized. KL divergence between
Venice's (likely quantized) deployment and our reference accounts for
this via calibration — we measure "normal" divergence first, then flag
anything that exceeds it. We can distinguish model classes (3B vs 70B)
but not specific quantization levels (FP16 vs INT4 of the same model).

### Data Provenance
The zkML proofs verify that our analysis pipeline ran correctly. They
do not independently prove the inputs came from Venice's API. The
mitigation is reproducibility: we publish our scripts and methodology
so anyone can re-run the same tests and compare results.

### Adversarial Robustness
Venice could theoretically detect audit-pattern queries and serve the
real model only for those. We mitigate with randomized prompts, varied
seeds, and queries that look like normal traffic — but cannot fully
eliminate this risk without continuous, opaque monitoring.

## Recommendations for Venice

To move from "externally testable" to "fully provable":
1. **Activate TEE attestation** — the `supportsTeeAttestation` field
   already exists in your model spec
2. **Integrate zkML proofs** into the inference pipeline so every
   response carries a model-identity proof
3. **Open-source the proxy layer** for independent code review
4. **Publish transparency reports** with infrastructure audit results
