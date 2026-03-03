# VerifyVenice Methodology

## Overview

Venice.ai claims uncensored inference and model authenticity. This
project tests those claims empirically and makes the analysis
verifiably honest via zkML proofs.

## What We're Testing

1. **Output Integrity (Group A):** Does Venice modify, filter, or censor
   model output? We compare Venice's responses against the same model
   served by independent providers. If Venice filters, the divergence
   is measurably abnormal.
2. **Model Authenticity (Group B):** Does Venice serve the model it claims
   (e.g., Llama 3.3 70B), or a cheaper substitute? We fingerprint
   logprob distributions — different model sizes produce measurably
   different patterns.

## Out of Scope

- Server-side data handling (requires TEEs or server integration)
- Decentralization claims (requires infrastructure audit)
- Network-level privacy (requires traffic analysis)

## Approach

### Group A: Output Integrity

For each of 5 prompt types (factual, reasoning, creative, code,
multilingual), we send identical prompts to:

- **Venice.ai** (with `include_venice_system_prompt: false`)
- **Together.ai** (as 70B reference)
- **Ollama** (as 3B local ground truth, where available)

Each prompt is repeated with 10 different seeds to capture distribution
properties rather than relying on single-point comparisons.

We compare using:
- **KL divergence** between top-5 logprob distributions
- **Token agreement rate** (fraction of positions where top-1 token matches)
- **Cosine similarity** of response text
- **Shannon entropy** of per-token logprob distributions

### Group B: Model Authenticity

We extract an 8-dimensional feature vector from each response's logprobs:
1. Mean top-1 logprob
2. Std of top-1 logprob
3. Mean entropy (from top-5 logprobs)
4. Mean gap between top-1 and top-2 logprob
5. Fraction of tokens where top-1 prob > 0.9
6. Mean perplexity
7. Vocabulary coverage in top-5
8. Mean spread of top-5 logprobs

A logistic regression classifier is trained on Together.ai 70B (class 0)
vs Venice 3B (class 1) features. The classifier is then applied to
Venice 70B responses to determine if they cluster with the 70B or 3B class.

### Calibration

Before comparing Venice against references, we establish a nondeterminism
baseline by running the same prompt 20 times on Ollama with identical
parameters. This measures:
- Expected KL divergence for the same model on the same hardware
- Expected token agreement rate for deterministic inference
- Entropy characteristics of the local model

These baselines define what "normal" divergence looks like, so we can
distinguish infrastructure differences from actual model differences.

### Additional Tests

- **System Prompt Differential:** Compare Venice responses with
  `include_venice_system_prompt: true` vs `false` to measure the
  system prompt's impact
- **Temporal Consistency:** Repeat the same query at different times
  to detect inconsistent model routing
- **Adversarial Probes:** Randomized prompts with varied seeds to
  make audit queries harder to fingerprint

### zkML Proofs

Two JOLT-Atlas circuits make the analysis verifiably honest:

1. **Output Comparison Circuit:** Proves the response comparison was
   computed correctly — nobody has to trust our similarity scores.

2. **Model Fingerprint Circuit:** Proves the classifier inference was
   computed correctly — the model-class prediction is verifiable.

The proofs guarantee analysis integrity. The remaining trust gap is
data provenance (did we faithfully capture the API responses?), which
is mitigated by publishing scripts so anyone can reproduce the tests.

## Statistical Thresholds

| Metric | Warn | Fail | Source |
|--------|------|------|--------|
| KL divergence | > 0.5 | > 2.0 | Empirical (calibration) |
| Token agreement | < 0.60 | < 0.30 | Empirical |
| Entropy ratio | < 0.5 or > 2.0 | < 0.25 or > 4.0 | Literature |
| Cosine similarity | < 0.85 | < 0.60 | Empirical |

## Sample Size

~50 calls per test group (5 prompt types x 10 repetitions). This is
sufficient for a proof-of-concept but NOT for audit-grade statistical
claims. Increasing to 100+ per group would strengthen conclusions.
