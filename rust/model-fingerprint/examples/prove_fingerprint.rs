//! CLI example: generate a model fingerprint proof.
//!
//! Usage:
//!   cargo run --release --example prove_fingerprint -- --input input.json [--model path.onnx] [--output-dir dir]

use circuits_common::save_proof;
use model_fingerprint::{prove, verify, FingerprintInput, CLASS_NAMES};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let input_path = args
        .iter()
        .position(|a| a == "--input")
        .map(|i| &args[i + 1])
        .unwrap_or_else(|| {
            eprintln!("Usage: prove_fingerprint --input <input.json> [--model <path.onnx>] [--output-dir <dir>]");
            std::process::exit(1);
        });

    let output_dir = args
        .iter()
        .position(|a| a == "--output-dir")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("../../data/proofs"));

    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| "models/model_fingerprint.onnx".to_string());

    // Load input
    let input_json = std::fs::read_to_string(input_path).expect("Failed to read input file");
    let input: FingerprintInput =
        serde_json::from_str(&input_json).expect("Failed to parse input JSON");

    println!("Generating model fingerprint proof...");
    println!("  Model: {}", input.model_name);
    println!("  Test ID: {}", input.test_id);
    println!("  Features: {:?}", input.features);
    println!("  ONNX model: {}", model_path);

    let start = std::time::Instant::now();
    let artifact = prove(&input, &model_path).expect("Proof generation failed");
    let elapsed = start.elapsed();

    println!("Proof generated in {:.2}s", elapsed.as_secs_f64());
    println!("  Proof size: {} bytes", artifact.proof_size_bytes);

    save_proof(&artifact, &output_dir).expect("Failed to save proof");

    match verify(&artifact, &model_path) {
        Ok(true) => println!("Verification: PASSED"),
        Ok(false) => println!("Verification: FAILED"),
        Err(e) => println!("Verification error: {}", e),
    }
}
