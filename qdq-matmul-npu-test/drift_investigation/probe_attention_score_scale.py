#!/usr/bin/env python3
"""Compare an EP attention result with exact-QDQ causal and score-scale references."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import onnx_ir as ir

from drift_investigation.qdq_ablation import QdqPair, attention_core_groups
from run_acc import file_sha256


@dataclass(frozen=True)
class QuantParams:
    scale: np.float32
    zero_point: int
    dtype: np.dtype


def quant_params(pair: QdqPair) -> QuantParams:
    inputs = pair.quantize.inputs
    if len(inputs) != 3 or inputs[1] is None or inputs[2] is None:
        raise ValueError(f"{pair.name}: explicit scalar QDQ parameters are required")
    if inputs[1].const_value is None or inputs[2].const_value is None:
        raise ValueError(f"{pair.name}: QDQ parameters must be constant")
    scale = inputs[1].const_value.numpy()
    zero_point = inputs[2].const_value.numpy()
    if scale.shape != () or zero_point.shape != () or scale.dtype != np.float32:
        raise ValueError(f"{pair.name}: expected scalar float32 scale and integer zero point")
    if zero_point.dtype not in (np.uint8, np.int8, np.uint16, np.int16):
        raise ValueError(f"{pair.name}: unsupported quantized data type {zero_point.dtype}")
    if not np.isfinite(scale).all() or scale.item() <= 0:
        raise ValueError(f"{pair.name}: QDQ scale or zero-point type is invalid")
    return QuantParams(np.float32(scale.item()), int(zero_point.item()), zero_point.dtype)


def apply_qdq(values: np.ndarray, params: QuantParams) -> np.ndarray:
    bounds = np.iinfo(params.dtype)
    quantized = np.clip(
        np.rint(values.astype(np.float64) / float(params.scale) + params.zero_point),
        bounds.min, bounds.max,
    ).astype(params.dtype)
    return ((quantized.astype(np.int32) - params.zero_point) * params.scale).astype(np.float32)


def attention_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray,
    indices: np.ndarray,
    scale_factor: float,
    parameters: tuple[QuantParams, QuantParams, QuantParams],
    *,
    causal: bool = False,
    causal_bias: float = -100.0,
) -> np.ndarray:
    score_params, masked_params, probability_params = parameters
    scores = apply_qdq((query[:, :, indices, :] @ key) * np.float32(scale_factor), score_params)
    bias = mask
    if causal:
        future = np.arange(key.shape[-1])[None, :] > indices[:, None]
        bias = bias + np.where(future, causal_bias, 0).astype(np.float32)[None, None, :, :]
    scores = apply_qdq(scores + bias, masked_params)
    probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return apply_qdq(probabilities, probability_params) @ value


def probe(
    model: ir.Model,
    inputs: dict[str, np.ndarray],
    ep_context: np.ndarray,
    factors: list[float],
    samples: int,
    cpu_reference: np.ndarray | None = None,
    causal_bias: float = -100.0,
) -> dict:
    groups = attention_core_groups(model.graph)
    if len(groups) != 1:
        raise ValueError(f"expected exactly one attention score pattern, found {len(groups)}")
    _, score_pair, masked_pair, probability_pair = groups[0]
    params = (
        quant_params(score_pair),
        quant_params(masked_pair),
        quant_params(probability_pair),
    )
    if not np.isfinite(factors).all() or any(factor <= 0 for factor in factors) or 1.0 not in factors:
        raise ValueError("factors must be finite, positive, and include 1.0")
    if samples <= 0:
        raise ValueError("samples must be positive")
    query, key, value, mask = (inputs[name] for name in ("query", "key", "value", "attention_mask"))
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or mask.ndim != 4:
        raise ValueError("query, key, value, and attention_mask must be rank-4 tensors")
    if (query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1] or
        query.shape[-1] != key.shape[-2] or query.shape[-2] != key.shape[-1] or
        value.shape[:3] != query.shape[:3] or mask.shape[-1] != key.shape[-1]):
        raise ValueError("attention inputs have incompatible shapes")
    if ep_context.shape != (query.shape[0], query.shape[1], query.shape[2], value.shape[-1]):
        raise ValueError("EP attention result shape does not match Q/K/V inputs")
    if any(tensor.dtype != np.float32 for tensor in (query, key, value, mask)):
        raise ValueError("this QDQ reference requires float32 Q/K/V/mask inputs")
    if any(not np.isfinite(tensor).all() for tensor in (query, key, value, mask, ep_context)):
        raise ValueError("EP output or attention inputs contain non-finite values")

    indices = np.unique(np.linspace(0, query.shape[2] - 1, min(samples, query.shape[2]), dtype=np.int64))
    target = ep_context[:, :, indices, :]
    output = {}
    for factor in factors:
        predicted = attention_reference(query, key, value, mask, indices, factor, params)
        output[str(factor)] = float(np.mean(np.abs(target - predicted)))
    baseline = attention_reference(query, key, value, mask, indices, 1.0, params)
    causal = attention_reference(query, key, value, mask, indices, 1.0, params, causal=True,
                                 causal_bias=causal_bias)
    cpu_mae = None
    if cpu_reference is not None:
        if cpu_reference.shape != ep_context.shape:
            raise ValueError("CPU reference and EP result shapes differ")
        cpu_mae = float(np.mean(np.abs(cpu_reference[:, :, indices, :] - baseline)))
        if cpu_mae > 1e-4:
            raise ValueError(f"NumPy QDQ reference does not match ORT CPU (MAE {cpu_mae})")

    causal_mae = np.mean(np.abs(target - causal), axis=(0, 1, 3))
    noncausal_mae = np.mean(np.abs(target - baseline), axis=(0, 1, 3))
    return {
        "indices": indices.tolist(),
        "factor_mae": output,
        "best_factor": min(factors, key=lambda factor: output[str(factor)]),
        "cpu_reference_mae": cpu_mae,
        "causal_mae": float(np.mean(causal_mae)),
        "noncausal_mae": float(np.mean(noncausal_mae)),
        "first_query_causal_mae": float(causal_mae[0]),
        "first_query_noncausal_mae": float(noncausal_mae[0]),
        "last_query_causal_mae": float(causal_mae[-1]),
        "last_query_noncausal_mae": float(noncausal_mae[-1]),
        "last_query_causal_effect": float(np.mean(np.abs(causal[:, :, -1, :] - baseline[:, :, -1, :])))
        if indices[-1] == query.shape[2] - 1 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Standalone QDQ attention island ONNX model.")
    parser.add_argument("--inputs", type=Path, required=True, help="Captured EP query/key/value/mask .npz.")
    parser.add_argument("--ep-output", type=Path, required=True, help=".npz with attention_context output.")
    parser.add_argument("--cpu-outputs", type=Path, help="Optional run_acc outputs .npz with cpu_0.")
    parser.add_argument("--samples", type=int, default=64, help="Evenly spaced query positions; default: 64.")
    parser.add_argument(
        "--factors", type=float, nargs="+",
        default=[0.125, 0.25, 0.45, 0.48, 0.49, 0.5, 0.505, 0.51, 0.52, 0.55, 1.0],
    )
    parser.add_argument("--report-json", type=Path, help="New machine-readable probe output.")
    args = parser.parse_args()
    if args.report_json is not None and args.report_json.exists():
        raise ValueError(f"report already exists: {args.report_json}")
    with np.load(args.inputs, allow_pickle=False) as archive:
        inputs = {name: archive[name] for name in ("query", "key", "value", "attention_mask")}
    with np.load(args.ep_output, allow_pickle=False) as archive:
        ep_context = archive["attention_context"]
    cpu_reference = None
    if args.cpu_outputs:
        with np.load(args.cpu_outputs, allow_pickle=False) as archive:
            cpu_reference = archive["cpu_0"]
    result = probe(ir.load(args.model), inputs, ep_context, args.factors, args.samples, cpu_reference)
    print(f"NumPy vs ORT CPU MAE: {result['cpu_reference_mae']}")
    for factor, mae in result["factor_mae"].items():
        print(f"QK score factor {factor:>6}: EP MAE {mae:.7f}")
    print(f"Causal MAE {result['causal_mae']:.7f}; noncausal MAE {result['noncausal_mae']:.7f}")
    print(
        f"Last query: causal MAE {result['last_query_causal_mae']:.7f}, "
        f"noncausal MAE {result['last_query_noncausal_mae']:.7f}, "
        f"causal reference change {result['last_query_causal_effect']:.7f}"
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps({
                **result,
                "model_sha256": file_sha256(args.model),
                "inputs_sha256": file_sha256(args.inputs),
                "ep_output_sha256": file_sha256(args.ep_output),
            }, indent=2) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
