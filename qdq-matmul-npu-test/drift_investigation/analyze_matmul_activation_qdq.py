#!/usr/bin/env python3
"""Measure activation-QDQ CPU and NPU effects across weight quantization schemes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnx_ir as ir
from onnx import numpy_helper

from compare_qdq_reports import compare_reports, read_report
from generate_qdq_matmul_test_suite import CATEGORIES, VARIANTS, Category, Shape, Variant, generate_model, run_command
from drift_investigation.qdq_ablation import activation_pairs, bypass_pairs, write_variant
from drift_investigation.study_resume import completed, next_attempt_dir
from run_acc import file_sha256


ACTIVATION_TYPES = ("uint16", "uint8")
REGIMES = ("input_clipped", "output_clipped")
WEIGHT_VARIANTS = ("int4_symmetric_no_zp", "uint8_asymmetric")
SHAPES = {
    "small": Shape("[1,128,128], [128,128]", "small", (1, 128, 128), (128, 128)),
    "vision": Shape("[1,2520,768], [768,768]", "vision", (1, 2520, 768), (768, 768)),
}


@dataclass(frozen=True)
class Case:
    category: Category
    variant: Variant
    activation_type: str
    regime: str

    @property
    def name(self) -> str:
        return f"{self.category.slug}_{self.variant.slug}_{self.activation_type}_{self.regime}"


def cases(categories: list[str], variants: list[str], activations: list[str], regimes: list[str]) -> list[Case]:
    category_map = {category.slug: category for category in CATEGORIES}
    variant_map = {variant.slug: variant for variant in VARIANTS}
    return [
        Case(category_map[category], variant_map[variant], activation, regime)
        for category in categories for variant in variants
        for activation in activations for regime in regimes
    ]


def activation_scales(activation_type: str, regime: str) -> tuple[float | None, float | None]:
    if activation_type not in ACTIVATION_TYPES or regime not in REGIMES:
        raise ValueError(f"unsupported activation type or stress regime: {activation_type}, {regime}")
    factor = 257 if activation_type == "uint8" else 1
    if regime == "input_clipped":
        return 0.00002 * factor, None
    return None, 0.000015 * factor


def out_of_range(values: np.ndarray, scale: float, zero_point: np.ndarray) -> float:
    limit = np.iinfo(zero_point.dtype)
    zero = int(zero_point.item())
    return float(np.mean((values < (limit.min - zero) * scale) | (values > (limit.max - zero) * scale)))


def quantization_parameters(model: Path, regime: str) -> tuple[float, np.ndarray]:
    proto = onnx.load(model, load_external_data=False)
    initializers = {value.name: numpy_helper.to_array(value) for value in proto.graph.initializer}
    name = "activation" if regime == "input_clipped" else "output"
    return float(initializers[name + "_scale"].item()), initializers[name + "_zero_point"]


def run_accuracy(
    python: str, runner: Path, model: Path, input_path: Path,
    context: Path, report: Path, arrays: Path,
    provider: str, provider_options: list[str],
) -> None:
    command = [
        python, str(runner), str(model),
        "--provider", provider,
        "--cpu-optimization", "basic",
        "--inputs", str(input_path),
        "--no-cpu-fallback",
        "--compile-output", str(context),
        "--report-json", str(report),
        "--output-arrays", str(arrays),
    ]
    for option in provider_options:
        command.extend(["--provider-option", option])
    code, output = run_command(command)
    report.with_suffix(".log").write_text(output, encoding="utf-8")
    if code:
        raise RuntimeError(f"{model.name}: CPU/NPU run failed (exit {code}): {output[-1500:]}")


def run_case(
    case: Case, output_dir: Path, input_path: Path, shape: Shape,
    args: argparse.Namespace,
) -> dict:
    folder = next_attempt_dir(output_dir, case.name)
    original, variant = folder / "original.onnx", folder / "activation_bypassed.onnx"
    activation_scale, output_scale = activation_scales(case.activation_type, case.regime)
    code, output = generate_model(
        sys.executable, Path(__file__).resolve().parent.parent / "generate_qdq_matmul_model.py",
        original, case.category, case.variant, shape,
        args.block_size, args.block_axis, args.seed,
        activation_type=case.activation_type,
        activation_scale=activation_scale,
        output_scale=output_scale,
    )
    (folder / "generator.log").write_text(output, encoding="utf-8")
    if code:
        raise RuntimeError(f"{case.name}: model generation failed (exit {code}): {output[-1500:]}")

    model = ir.load(original)
    all_pairs = activation_pairs(model.graph)
    chosen_name = "QuantizeActivation" if case.regime == "input_clipped" else "QuantizeOutput"
    selected = [pair for pair in all_pairs if pair.name == chosen_name]
    if len(all_pairs) != 2 or len(selected) != 1:
        raise ValueError(f"{case.name}: expected exactly two activation QDQ pairs and one {chosen_name}")
    bypass_pairs(model, selected)
    write_variant(
        model, original, variant, 1,
        {"mode": "exact", "selected_pairs": [chosen_name], "regime": case.regime},
    )
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
    comparison = compare_reports(read_report(folder / "original.json"), read_report(folder / "variant.json"))
    metrics = comparison["outputs"]["output"]
    with np.load(input_path, allow_pickle=False) as input_archive:
        input_value = input_archive["input"]
    with np.load(folder / "original_outputs.npz", allow_pickle=False) as baseline:
        npu_original = baseline["ep_0"]
    with np.load(folder / "variant_outputs.npz", allow_pickle=False) as ablated:
        npu_variant, cpu_variant = ablated["ep_0"], ablated["cpu_0"]
    scale, zero_point = quantization_parameters(original, case.regime)
    stress_fraction = out_of_range(input_value if case.regime == "input_clipped" else cpu_variant, scale, zero_point)
    if stress_fraction < 0.05 or metrics["cpu_shift_mae"] < 0.001:
        raise ValueError(
            f"{case.name}: QDQ stress was too weak (outside range {stress_fraction:.2%}, "
            f"CPU shift {metrics['cpu_shift_mae']:.5g}); result cannot diagnose stripping"
        )
    result = {
        "category": case.category.slug,
        "weight_quantization": case.category.quantization,
        "weight_variant": case.variant.slug,
        "qdq_profile": case.category.qdq_profile,
        "activation_type": case.activation_type,
        "regime": case.regime,
        "model_shape": shape.slug,
        "stress_fraction": stress_fraction,
        "cpu_shift_mae": metrics["cpu_shift_mae"],
        "npu_shift_mae": metrics["ep_shift_mae"],
        "npu_bitwise_unchanged": bool(np.array_equal(npu_original, npu_variant)),
        "npu_original_outside_output_qdq_range": (
            out_of_range(npu_original, *quantization_parameters(original, "output_clipped"))
        ),
        "original_cpu_npu_mae": metrics["baseline_ep_cpu_mae"],
        "variant_cpu_npu_mae": metrics["variant_ep_cpu_mae"],
        "original_cpu_npu_cosine": metrics["baseline_ep_cpu_cosine"],
        "variant_cpu_npu_cosine": metrics["variant_ep_cpu_cosine"],
        "input_sha256": file_sha256(input_path),
    }
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "unit-models" / "activation-qdq-study" / "matrix",
    )
    parser.add_argument("--provider", choices=("openvino", "qnn"), default="openvino")
    parser.add_argument("--provider-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--category", action="append", choices=[category.slug for category in CATEGORIES])
    weight_selection = parser.add_mutually_exclusive_group()
    weight_selection.add_argument("--weight-variant", action="append", choices=[variant.slug for variant in VARIANTS])
    weight_selection.add_argument(
        "--all-weight-variants", action="store_true",
        help="Run all ten weight types from the existing MatMul suite instead of two representative defaults.",
    )
    parser.add_argument("--activation-type", action="append", choices=ACTIVATION_TYPES)
    parser.add_argument("--regime", action="append", choices=REGIMES)
    parser.add_argument("--shape", choices=tuple(SHAPES), default="small")
    parser.add_argument("--block-size", type=int, choices=(32, 128), default=32)
    parser.add_argument("--block-axis", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--input-seed", type=int, default=20260922)
    parser.add_argument("--limit", type=int, help="Run only the first N cases; resume to complete the matrix.")
    parser.add_argument("--resume", action="store_true", help="Skip successful cases and retry failed cases.")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    shape = SHAPES[args.shape]
    weight_variants = (
        [variant.slug for variant in VARIANTS]
        if args.all_weight_variants else (args.weight_variant or list(WEIGHT_VARIANTS))
    )
    selected = cases(
        args.category or [category.slug for category in CATEGORIES],
        weight_variants,
        args.activation_type or list(ACTIVATION_TYPES),
        args.regime or list(REGIMES),
    )
    config = {
        "provider": args.provider, "provider_options": args.provider_option,
        "shape": args.shape, "block_size": args.block_size, "block_axis": args.block_axis,
        "seed": args.seed, "input_seed": args.input_seed,
        "cases": [case.name for case in selected],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    input_path = output_dir / "inputs.npz"
    if summary_path.exists():
        if not args.resume:
            raise ValueError(f"study already exists; use --resume: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["config"] != config:
            raise ValueError("resume parameters differ from the existing study configuration")
        if not input_path.is_file() or file_sha256(input_path) != summary["inputs_sha256"]:
            raise ValueError("saved study inputs are absent or changed")
    else:
        if args.resume or input_path.exists():
            raise ValueError("cannot resume a nonexistent study or overwrite preexisting inputs")
        values = np.random.default_rng(args.input_seed).standard_normal(shape.input_shape).astype(np.float32)
        np.savez_compressed(input_path, input=np.clip(values, -4, 4))
        summary = {"config": config, "inputs_sha256": file_sha256(input_path), "results": {}}
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for index, case in enumerate(selected[: args.limit], start=1):
        if completed(summary["results"].get(case.name)):
            print(f"[{index}] already recorded: {case.name}", flush=True)
            continue
        print(f"[{index}/{len(selected)}] {case.name}", flush=True)
        try:
            result = run_case(case, output_dir, input_path, shape, args)
        except (RuntimeError, ValueError) as error:
            result = {"error": str(error)}
        summary["results"][case.name] = result
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if "error" in result:
            print(f"  ERROR: {result['error']}", flush=True)
        else:
            print(
                f"  stress={result['stress_fraction']:.1%} CPU shift={result['cpu_shift_mae']:.5g}, "
                f"NPU shift={result['npu_shift_mae']:.5g}, invariant={result['npu_bitwise_unchanged']}",
                flush=True,
            )
    failures = [name for name, value in summary["results"].items() if "error" in value]
    print(f"Saved {len(summary['results'])}/{len(selected)} cases to {summary_path}; failures: {len(failures)}")
    if failures:
        raise RuntimeError(f"study has failed cases: {failures}")


if __name__ == "__main__":
    main()
