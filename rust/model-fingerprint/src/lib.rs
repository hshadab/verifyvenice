//! Circuit 2: Model Fingerprint Classification Proof
//!
//! Proves that the auditor correctly classified a logprob feature
//! vector using the trained fingerprint classifier. The proof attests:
//! "Given this 8-dimensional feature vector, the classifier produced
//! these class probabilities."
//!
//! Classes: [0 = 70b-class, 1 = 3b-class, 2 = unknown]

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
const N_FEATURES: usize = 8;

/// Class labels for the fingerprint classifier.
pub const CLASS_NAMES: [&str; 3] = ["70b-class", "3b-class", "unknown"];

/// Input data for the model fingerprint circuit.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FingerprintInput {
    /// 8-dimensional feature vector from logprob analysis
    pub features: Vec<f64>,
    /// Model name being tested (for metadata)
    pub model_name: String,
    /// Test ID (for traceability)
    pub test_id: String,
}

impl FingerprintInput {
    pub fn to_fixed_point(&self) -> Vec<i32> {
        self.features
            .iter()
            .map(|&v| float_to_fixed_i32(v, SCALE_BITS))
            .collect()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.features.len() != N_FEATURES {
            return Err(format!(
                "Expected {} features, got {}",
                N_FEATURES,
                self.features.len()
            ));
        }
        Ok(())
    }
}

/// Generate a proof for the fingerprint classification.
pub fn prove(input: &FingerprintInput, model_path: &str) -> Result<ProofArtifact, String> {
    input.validate()?;
    let fixed_input = input.to_fixed_point();

    // Load ONNX model
    let model = Model::load(model_path, &RunArgs::default());

    // Preprocessing
    let shared = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(shared);

    // Create input tensor: shape [1, 8]
    let tensor = Tensor::new(Some(&fixed_input), &[1, N_FEATURES])
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

    // Store the input features as IO bytes (verifier will re-derive ModelExecutionIO)
    let io_bytes = serde_json::to_vec(input)
        .map_err(|e| format!("Failed to serialize input: {}", e))?;

    let mut artifact = ProofArtifact::new("model_fingerprint", proof_bytes, io_bytes, vec![]);
    artifact.input_hash = format!("test_id={},model={}", input.test_id, input.model_name);
    Ok(artifact)
}

/// Verify a previously generated proof.
pub fn verify(artifact: &ProofArtifact, model_path: &str) -> Result<bool, String> {
    let proof = Proof::deserialize_compressed(&artifact.proof_bytes[..])
        .map_err(|e| format!("Failed to deserialize proof: {}", e))?;

    // Reconstruct input from stored bytes
    let input: FingerprintInput = serde_json::from_slice(&artifact.io_bytes)
        .map_err(|e| format!("Failed to deserialize input: {}", e))?;
    let fixed_input = input.to_fixed_point();

    // Rebuild model, preprocessing, and IO
    let model = Model::load(model_path, &RunArgs::default());
    let shared = AtlasSharedPreprocessing::preprocess(model);
    let prover_pp = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(shared);
    let verifier_pp = AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_pp);

    // Re-derive IO by running the model with the same inputs
    let tensor = Tensor::new(Some(&fixed_input), &[1, N_FEATURES])
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
    fn test_fingerprint_input_validation() {
        let valid = FingerprintInput {
            features: vec![0.0; 8],
            model_name: "test".to_string(),
            test_id: "t1".to_string(),
        };
        assert!(valid.validate().is_ok());

        let invalid = FingerprintInput {
            features: vec![0.0; 5],
            model_name: "test".to_string(),
            test_id: "t1".to_string(),
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_fixed_point_conversion() {
        let input = FingerprintInput {
            features: vec![1.0, -2.5, 0.5, 3.14, 0.0, 1.0, 100.0, 0.1],
            model_name: "test".to_string(),
            test_id: "t1".to_string(),
        };
        let fixed = input.to_fixed_point();
        assert_eq!(fixed.len(), 8);
        assert_eq!(fixed[0], 4096); // 1.0 * 2^12
    }
}
