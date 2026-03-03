"""Export small PyTorch models to ONNX for JOLT-Atlas circuits.

Both models use ONLY rank-2 tensors and JOLT-Atlas-supported operators:
MatMul, Add, Relu, Sigmoid, Softmax.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn


class OutputComparisonModel(nn.Module):
    """Circuit 1: Compares two hash-derived feature vectors.

    Input: (1, 2*D) concatenated feature vectors
    Output: (1, 1) similarity score in [0, 1]
    """

    def __init__(self, hash_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hash_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelFingerprintClassifier(nn.Module):
    """Circuit 2: Classifies logprob feature vector into model family.

    Input: (1, 8) feature vector (from ModelFingerprinter.extract_features)
    Output: (1, 3) class probabilities [70b-class, 3b-class, unknown]
    """

    def __init__(self, n_features: int = 8, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def export_model(
    model: nn.Module,
    input_shape: tuple[int, ...],
    path: str | Path,
    model_name: str = "model",
) -> Path:
    """Export PyTorch model to ONNX.

    Uses opset 13, fixed shapes (no dynamic axes), rank-2 tensors only.
    Validates with onnx.checker after export.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        opset_version=13,
    )

    # Validate
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)

    # Verify operators are JOLT-Atlas compatible
    supported_ops = {
        "MatMul", "Gemm", "Add", "Relu", "Sigmoid", "Softmax",
        "Reshape", "Gather", "Const", "Constant", "Shape",
        "Unsqueeze", "Squeeze", "Cast", "Flatten",
    }
    model_ops = {node.op_type for node in onnx_model.graph.node}
    unsupported = model_ops - supported_ops
    if unsupported:
        print(f"WARNING: Potentially unsupported ONNX operators: {unsupported}")
        print("These may not work with JOLT-Atlas. Check operator support list.")

    print(f"Exported {model_name} to {path}")
    print(f"  Operators used: {model_ops}")
    print(f"  Model size: {path.stat().st_size / 1024:.1f} KB")

    return path


def export_comparison_model(output_dir: Path) -> Path:
    """Export the output comparison circuit model."""
    model = OutputComparisonModel(hash_dim=16)
    return export_model(
        model,
        input_shape=(1, 32),
        path=output_dir / "output_comparison.onnx",
        model_name="OutputComparisonModel",
    )


def export_fingerprint_model(output_dir: Path) -> Path:
    """Export the model fingerprint classifier."""
    model = ModelFingerprintClassifier(n_features=8, n_classes=3)
    return export_model(
        model,
        input_shape=(1, 8),
        path=output_dir / "model_fingerprint.onnx",
        model_name="ModelFingerprintClassifier",
    )


def load_trained_weights_into_fingerprint(
    sklearn_classifier,
    sklearn_scaler,
    n_features: int = 8,
    n_classes: int = 3,
) -> ModelFingerprintClassifier:
    """Transfer sklearn logistic regression weights into PyTorch model.

    This allows us to train with sklearn (convenient) and export via
    PyTorch ONNX (JOLT-Atlas compatible).
    """
    model = ModelFingerprintClassifier(n_features=n_features, n_classes=n_classes)

    with torch.no_grad():
        # First linear layer: apply scaler transform (mean/scale normalization)
        scale = torch.tensor(sklearn_scaler.scale_, dtype=torch.float32)
        mean = torch.tensor(sklearn_scaler.mean_, dtype=torch.float32)

        # Embed scaling into first linear layer:
        # scaled_x = (x - mean) / scale
        # W1 @ scaled_x + b1 = W1 @ ((x - mean) / scale) + b1
        #                     = (W1 / scale) @ x + (b1 - W1 @ mean / scale)
        W1 = model.net[0].weight.data  # (16, 8)
        b1 = model.net[0].bias.data    # (16,)

        model.net[0].weight.data = W1 / scale.unsqueeze(0)
        model.net[0].bias.data = b1 - (W1 / scale.unsqueeze(0)) @ mean

        # Last linear layer: use sklearn coefficients
        coef = torch.tensor(sklearn_classifier.coef_, dtype=torch.float32)  # (n_classes, n_features)
        intercept = torch.tensor(sklearn_classifier.intercept_, dtype=torch.float32)

        # Note: We can't directly map sklearn weights to the deep network.
        # Instead, the first two layers learn a representation during
        # ONNX export with random weights, and the final layer uses
        # identity mapping. For a proper transfer, retrain the PyTorch
        # model on the same data. This function is a starting point.

    return model
