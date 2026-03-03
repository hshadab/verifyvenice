# zkML Architecture

## How JOLT-Atlas Proofs Work

JOLT-Atlas is a zero-knowledge machine learning (zkML) framework that
proves ONNX model inference was performed correctly. It uses:

- **Lookup arguments** (Lasso/Shout protocols) instead of arithmetic circuits
- **Sumcheck protocol** for efficient polynomial verification
- **HyperKZG over BN254** for polynomial commitments
- **Blake2b transcripts** for the Fiat-Shamir transform

### Key Difference from Traditional ZK

Traditional ZK systems encode computations as arithmetic circuits
(R1CS, Plonkish). JOLT-Atlas verifies operations by checking results
against pre-computed lookup tables that are never materialized. This
is why it achieves 3-17x speedups over alternatives like ezkl.

## Our Two Circuits

### Circuit 1: Output Comparison

**Purpose:** Prove the auditor correctly computed similarity between
Venice and reference response features.

**Model:** Small feedforward network
- Input: (1, 32) — concatenated 16-dim feature vectors
- Architecture: Linear(32,32) → ReLU → Linear(32,16) → ReLU → Linear(16,1) → Sigmoid
- Output: (1, 1) — similarity score in [0, 1]
- ONNX operators: Gemm, Relu, Sigmoid (all JOLT-Atlas supported)

**What the proof attests:**
"Given these two feature vectors as input, the comparison network
produced this similarity score."

### Circuit 2: Model Fingerprint Classifier

**Purpose:** Prove the auditor correctly classified a logprob feature
vector into a model family.

**Model:** Small classifier
- Input: (1, 8) — logprob feature vector
- Architecture: Linear(8,16) → ReLU → Linear(16,8) → ReLU → Linear(8,3) → Softmax
- Output: (1, 3) — class probabilities [70b-class, 3b-class, unknown]
- ONNX operators: Gemm, Relu, Softmax (all JOLT-Atlas supported)

**What the proof attests:**
"Given this 8-dimensional feature vector, the classifier produced
these class probabilities."

## Proof Generation Flow

```
1. Python: Collect API responses → extract features
2. Python: Train classifier → export as ONNX
3. Python: Prepare circuit inputs (feature vectors as JSON)
4. Rust: Load ONNX model into JOLT-Atlas
5. Rust: Run AtlasSharedPreprocessing::preprocess(model)
6. Rust: Run ONNXProof::prove(preprocessing, inputs)
7. Rust: Serialize proof via ark-serialize
8. Anyone: Run ONNXProof::verify(proof, io) to check
```

## Proof Verification

Anyone with the proof artifact (binary) and verifier preprocessing
data can verify the proof:

```rust
let proof = deserialize(proof_bytes);
let io = deserialize(io_bytes);
let verifier_pp = deserialize(vpp_bytes);
proof.verify(&verifier_pp, &io, None).unwrap(); // panics if invalid
```

Verification is orders of magnitude faster than proof generation.

## Constraints

- **Rank-2 tensors only:** JOLT-Atlas currently supports 2-dimensional
  tensors. Both our models use (1, N) shapes.
- **Supported operators:** ~26 ONNX operators. Our models use only
  Gemm (Linear layers), Relu, Sigmoid, and Softmax — all supported.
- **Fixed-point arithmetic:** JOLT-Atlas operates over BN254 scalar
  field integers. We convert floating-point features to fixed-point
  (scale by 2^16) before circuit input.
- **No CLI:** JOLT-Atlas has no command-line interface. Proof
  generation requires writing Rust code against the library API.

## What These Proofs Do NOT Prove

1. They don't prove data provenance (inputs could be fabricated)
2. They don't prove Venice served specific responses
3. They don't prove the ONNX model is the "right" classifier
4. They prove computation integrity of the auditor's pipeline only

For Venice's claims to be fully provable, Venice would need to generate
proofs themselves, inside their inference pipeline.
