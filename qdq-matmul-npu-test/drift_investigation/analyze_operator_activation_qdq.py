#!/usr/bin/env python3
"""Compare CPU and NPU activation-QDQ effects around Gemma-style weight-free operators."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnx_ir as ir

from drift_investigation.analyze_matmul_activation_qdq import out_of_range, quantization_parameters, run_accuracy
from compare_qdq_reports import compare_reports, read_report
from drift_investigation.generate_qdq_operator_model import ACTIVATION_TYPES, OPERATORS, REGIMES, build_model
from drift_investigation.qdq_ablation import activation_pairs, bypass_pairs, write_variant
from drift_investigation.study_resume import completed, next_attempt_dir
from run_acc import file_sha256, measure_float_output


def source_operator_counts(path: Path, selected_ops: list[str]) -> dict[str, int]:
    graph = ir.load(path).graph
    counts = Counter(
        pair.quantize.inputs[0].producer().op_type
        for pair in activation_pairs(graph)
        if pair.quantize.inputs[0].producer() is not None
    )
    missing = [name for name in selected_ops if counts[name] == 0]
    if missing:
        raise ValueError(f"reference model has no QDQ-wrapped operators: {missing}")
    return {name: counts[name] for name in selected_ops}


def run_case(
    output_dir: Path, op_type: str, activation_type: str, regime: str, args: argparse.Namespace,
) -> dict:
    name = f"{op_type.lower()}_{activation_type}_{regime}"
    folder = next_attempt_dir(output_dir, name)
    original = folder / "original.onnx"
    variant = folder / "activation_bypassed.onnx"
    model, inputs = build_model(op_type, activation_type, regime, args.seed)
    onnx.save(model, original)
    input_path = folder / "inputs.npz"
    np.savez_compressed(input_path, **inputs)

    graph = ir.load(original)
    pairs = activation_pairs(graph.graph)
    target = "QuantizeActivation" if regime == "input_clipped" else "QuantizeOutput"
    selected = [pair for pair in pairs if pair.name == target]
    if len(pairs) != 2 or len(selected) != 1:
        raise ValueError(f"{name}: expected two activation pairs, including {target}")
    bypass_pairs(graph, selected)
    write_variant(graph, original, variant, 1, {"mode": "exact", "selected_pairs": [target]})
    runner = Path(__file__).resolve().parent.parent / "run_acc.py"
    run_accuracy(
        sys.executable, runner, original, input_path,
        folder / "original_ctx.onnx", folder / "original.json", folder / "original_outputs.npz",
        args.provider, args.provider_option,
    )
    run_accuracy(
        sys.executable, runner, variant, input_path,
        folder / "variant_ctx.onnx", folder / "variant.json", folder / "variant_outputs.npz",
        args.provider, args.provider_option,
    )
    metrics = compare_reports(read_report(folder / "original.json"), read_report(folder / "variant.json"))[
        "outputs"]["output"]
    with np.load(folder / "original_outputs.npz", allow_pickle=False) as archive:
        original_cpu, original_ep = archive["cpu_0"], archive["ep_0"]
    with np.load(folder / "variant_outputs.npz", allow_pickle=False) as archive:
        variant_cpu, variant_ep = archive["cpu_0"], archive["ep_0"]
    scale, zero_point = quantization_parameters(original, regime)
    clipped = out_of_range(inputs["x"] if regime == "input_clipped" else variant_cpu, scale, zero_point)
    if clipped < 0.05 or metrics["cpu_shift_mae"] < 0.001:
        raise ValueError(
            f"{name}: QDQ stress is not diagnostic (outside range {clipped:.2%}, "
            f"CPU shift {metrics['cpu_shift_mae']:.5g})"
        )
    cpu_delta = variant_cpu.astype(np.float64) - original_cpu.astype(np.float64)
    ep_delta = variant_ep.astype(np.float64) - original_ep.astype(np.float64)
    delta_metrics = measure_float_output(cpu_delta, ep_delta, 0, 0)
    result = {
        "operator": op_type,
        "activation_type": activation_type,
        "regime": regime,
        "cpu_optimization": "basic",
        "clipped_fraction": clipped,
        "cpu_shift_mae": metrics["cpu_shift_mae"],
        "npu_shift_mae": metrics["ep_shift_mae"],
        "npu_bitwise_unchanged": bool(np.array_equal(original_ep, variant_ep)),
        "delta_alignment_cosine": (
            delta_metrics["cosine_similarity"] if not np.array_equal(original_ep, variant_ep) else None
        ),
        "delta_disagreement_mae": delta_metrics["mean_abs_error"],
        "original_cpu_npu_mae": metrics["baseline_ep_cpu_mae"],
        "variant_cpu_npu_mae": metrics["variant_ep_cpu_mae"],
        "npu_outside_original_output_qdq_range": out_of_range(
            original_ep, *quantization_parameters(original, "output_clipped")
        ),
        "inputs_sha256": file_sha256(input_path),
    }
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "unit-models" / "operator-qdq-study" / "matrix",
    )
    parser.add_argument("--source-model", type=Path, help="Gemma encoder ONNX for validating source op coverage.")
    parser.add_argument("--op", action="append", choices=OPERATORS)
    parser.add_argument("--activation-type", action="append", choices=ACTIVATION_TYPES)
    parser.add_argument("--regime", action="append", choices=REGIMES)
    parser.add_argument("--provider", choices=("openvino", "qnn"), default="openvino")
    parser.add_argument("--provider-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--limit", type=int, help="Run only the first N cases and save resumable progress.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.source_model is not None and not args.source_model.is_file():
        parser.error(f"reference model does not exist: {args.source_model}")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ops = args.op or list(OPERATORS)
    activations = args.activation_type or list(ACTIVATION_TYPES)
    regimes = args.regime or list(REGIMES)
    selected = [(op, activation, regime) for op in ops for activation in activations for regime in regimes]
    reference = args.source_model.resolve() if args.source_model else None
    config = {
        "provider": args.provider,
        "provider_options": args.provider_option,
        "seed": args.seed,
        "source_model": str(reference) if reference else None,
        "source_sha256": file_sha256(reference) if reference else None,
        "operator_counts": source_operator_counts(reference, ops) if reference else None,
        "cases": [f"{op.lower()}_{activation}_{regime}" for op, activation, regime in selected],
    }
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        if not args.resume:
            raise ValueError(f"study already exists; use --resume: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["config"] != config:
            raise ValueError("resume parameters do not match the saved operator study")
    else:
        if args.resume:
            raise ValueError(f"cannot resume missing study: {summary_path}")
        summary = {"config": config, "results": {}}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for index, (op, activation, regime) in enumerate(selected[: args.limit], 1):
        name = f"{op.lower()}_{activation}_{regime}"
        if completed(summary["results"].get(name)):
            print(f"[{index}] already recorded: {name}", flush=True)
            continue
        print(f"[{index}/{len(selected)}] {name}", flush=True)
        try:
            result = run_case(output_dir, op, activation, regime, args)
        except (RuntimeError, ValueError) as error:
            result = {"error": str(error)}
        summary["results"][name] = result
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if "error" in result:
            print(f"  ERROR: {result['error']}", flush=True)
        else:
            print(
                f"  clipped={result['clipped_fraction']:.1%} CPU shift={result['cpu_shift_mae']:.5g}, "
                f"NPU shift={result['npu_shift_mae']:.5g}, invariant={result['npu_bitwise_unchanged']}",
                flush=True,
            )
    failures = [name for name, result in summary["results"].items() if "error" in result]
    print(f"Saved {len(summary['results'])}/{len(selected)} operator cases to {summary_path}; failures={len(failures)}")
    if failures:
        raise RuntimeError(f"operator study has failed cases: {failures}")


if __name__ == "__main__":
    main()
