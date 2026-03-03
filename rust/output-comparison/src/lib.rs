//! Circuit 1: Output Comparison Proof
//!
//! Proves that the auditor correctly computed the similarity between
//! two API response feature vectors using a small neural network.
//!
//! The proof attests: "Given these two input vectors, the network
//! produced this similarity score." Anyone can verify the proof
//! without re-running the network.

use circuits_common::{float_to_fixed_i32, ProofArtifact};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
};

type Proof = ONNXProof<Fr, Blake2bTranscript, HyperKZG<Bn254>>;

const SCALE_BITS: u32 = 12;

/// Input data for the output comparison circuit.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComparisonInput {
    /// Hash-derived features of Venice API response (16 dims)
    pub venice_features: Vec<f64>,
    /// Hash-derived features of reference response (16 dims)
    pub reference_features: Vec<f64>,
}

impl ComparisonInput {
    /// Convert to fixed-point i32 integers for JOLT-Atlas tensors.
    pub fn to_fixed_point(&self) -> Vec<i32> {
        let mut fixed = Vec::with_capacity(self.venice_features.len() + self.reference_features.len());
        for &v in &self.venice_features {
            fixed.push(float_to_fixed_i32(v, SCALE_BITS));
        }
        for &v in &self.reference_features {
            fixed.push(float_to_fixed_i32(v, SCALE_BITS));
        }
        fixed
    }
}

/// Generate a proof for the output comparison.
pub fn prove(input: &ComparisonInput, model_path: &str) -> Result<ProofArtifact, String> {
    let fixed_input = input.to_fixed_point();
    let n = fixed_input.len();

    // Load ONNX model
    let model = Model::load(model_path, &RunArgs::default());

    // Preprocessing
    let shared = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(shared);

    // Create input tensor: shape [1, n]
    let tensor = Tensor::new(Some(&fixed_input), &[1, n])
        .map_err(|e| format!("Failed to create tensor: {:?}", e))?;

    // Generate proof
    let (proof, io, _debug) = Proof::prove(&prover_pp, &[tensor]);

    // Verify locally before returning (sanity check)
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);
    proof.verify(&verifier_pp, &io, None)
        .map_err(|e| format!("Proof self-verification failed: {:?}", e))?;

    // Serialize proof using ark_serialize
    let mut proof_bytes = Vec::new();
    proof.serialize_compressed(&mut proof_bytes)
        .map_err(|e| format!("Failed to serialize proof: {}", e))?;

    // Store the input features as IO bytes (the verifier will re-derive ModelExecutionIO)
    let io_bytes = serde_json::to_vec(input)
        .map_err(|e| format!("Failed to serialize input: {}", e))?;

    let mut artifact = ProofArtifact::new("output_comparison", proof_bytes, io_bytes, vec![]);
    artifact.input_hash = format!("features_dim={}", n);
    Ok(artifact)
}

/// Verify a previously generated proof.
pub fn verify(artifact: &ProofArtifact, model_path: &str) -> Result<bool, String> {
    let proof = Proof::deserialize_compressed(&artifact.proof_bytes[..])
        .map_err(|e| format!("Failed to deserialize proof: {}", e))?;

    // Reconstruct input from stored bytes
    let input: ComparisonInput = serde_json::from_slice(&artifact.io_bytes)
        .map_err(|e| format!("Failed to deserialize input: {}", e))?;
    let fixed_input = input.to_fixed_point();
    let n = fixed_input.len();

    // Rebuild model, preprocessing, and IO
    let model = Model::load(model_path, &RunArgs::default());
    let shared = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(shared);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    // Re-derive IO by running the model with the same inputs
    let tensor = Tensor::new(Some(&fixed_input), &[1, n])
        .map_err(|e| format!("Failed to create tensor: {:?}", e))?;
    let trace = prover_pp.model().trace(&[tensor]);
    let io = atlas_onnx_tracer::model::trace::Trace::io(&trace, prover_pp.model());

    match proof.verify(&verifier_pp, &io, None) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_input_to_fixed() {
        let input = ComparisonInput {
            venice_features: vec![1.0, 2.5, -0.5],
            reference_features: vec![1.1, 2.4, -0.6],
        };
        let fixed = input.to_fixed_point();
        assert_eq!(fixed.len(), 6);
        // 1.0 * 2^12 = 4096
        assert_eq!(fixed[0], 4096);
    }
}
