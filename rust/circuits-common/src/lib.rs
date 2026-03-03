//! Shared types and utilities for VerifyVenice zkML circuits.
//!
//! Provides proof artifact serialization, field element conversion,
//! and I/O helpers used by both the output-comparison and
//! model-fingerprint circuits.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// A serialized proof artifact with metadata.
///
/// Contains the raw proof bytes, I/O bytes, verifier preprocessing,
/// and audit metadata. All byte fields are serialized via
/// `ark-serialize::CanonicalSerialize` in compressed mode.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProofArtifact {
    pub circuit_name: String,
    pub proof_bytes: Vec<u8>,
    pub io_bytes: Vec<u8>,
    pub verifier_preprocessing_bytes: Vec<u8>,
    pub timestamp: String,
    pub input_hash: String,
    pub proof_size_bytes: usize,
}

impl ProofArtifact {
    pub fn new(circuit_name: &str, proof_bytes: Vec<u8>, io_bytes: Vec<u8>, vpp_bytes: Vec<u8>) -> Self {
        let proof_size = proof_bytes.len();
        Self {
            circuit_name: circuit_name.to_string(),
            proof_bytes,
            io_bytes,
            verifier_preprocessing_bytes: vpp_bytes,
            timestamp: chrono::Utc::now().to_rfc3339(),
            input_hash: String::new(),
            proof_size_bytes: proof_size,
        }
    }
}

/// Save a proof artifact to disk as JSON + binary.
pub fn save_proof(artifact: &ProofArtifact, dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;

    let meta_path = dir.join(format!("{}_meta.json", artifact.circuit_name));
    let proof_path = dir.join(format!("{}_proof.bin", artifact.circuit_name));

    // Save metadata (everything except raw bytes)
    let meta = serde_json::json!({
        "circuit_name": artifact.circuit_name,
        "timestamp": artifact.timestamp,
        "input_hash": artifact.input_hash,
        "proof_size_bytes": artifact.proof_size_bytes,
        "io_size_bytes": artifact.io_bytes.len(),
        "vpp_size_bytes": artifact.verifier_preprocessing_bytes.len(),
    });
    fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;

    // Save proof bytes
    fs::write(&proof_path, &artifact.proof_bytes)?;

    println!("Proof saved to {}", dir.display());
    println!("  Metadata: {}", meta_path.display());
    println!("  Proof binary: {} ({} bytes)", proof_path.display(), artifact.proof_size_bytes);

    Ok(())
}

/// Load a proof artifact from disk.
pub fn load_proof(dir: &Path, circuit_name: &str) -> std::io::Result<ProofArtifact> {
    let meta_path = dir.join(format!("{}_meta.json", circuit_name));
    let proof_path = dir.join(format!("{}_proof.bin", circuit_name));

    let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(&meta_path)?)?;
    let proof_bytes = fs::read(&proof_path)?;

    Ok(ProofArtifact {
        circuit_name: meta["circuit_name"].as_str().unwrap_or("").to_string(),
        proof_bytes,
        io_bytes: Vec::new(),
        verifier_preprocessing_bytes: Vec::new(),
        timestamp: meta["timestamp"].as_str().unwrap_or("").to_string(),
        input_hash: meta["input_hash"].as_str().unwrap_or("").to_string(),
        proof_size_bytes: meta["proof_size_bytes"].as_u64().unwrap_or(0) as usize,
    })
}

/// Convert a floating-point value to fixed-point integer representation.
///
/// JOLT-Atlas operates over BN254 scalar field elements (integers).
/// We scale floats by 2^scale_bits and round to the nearest integer.
pub fn float_to_fixed(val: f64, scale_bits: u32) -> i64 {
    let scale = (1u64 << scale_bits) as f64;
    (val * scale).round() as i64
}

/// Convert float to i32 fixed-point for JOLT-Atlas tensors.
///
/// JOLT-Atlas Tensor uses i32 elements internally.
/// Uses 12 scale bits to stay safely within i32 range.
pub fn float_to_fixed_i32(val: f64, scale_bits: u32) -> i32 {
    let scale = (1u64 << scale_bits) as f64;
    (val * scale).round() as i32
}

/// Convert fixed-point integer back to floating-point.
pub fn fixed_to_float(val: i64, scale_bits: u32) -> f64 {
    let scale = (1u64 << scale_bits) as f64;
    val as f64 / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_roundtrip() {
        let val = 3.14159;
        let fixed = float_to_fixed(val, 16);
        let back = fixed_to_float(fixed, 16);
        assert!((val - back).abs() < 1e-4);
    }

    #[test]
    fn test_fixed_point_negative() {
        let val = -2.718;
        let fixed = float_to_fixed(val, 16);
        let back = fixed_to_float(fixed, 16);
        assert!((val - back).abs() < 1e-4);
    }
}
