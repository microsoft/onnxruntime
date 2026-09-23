#!/usr/bin/env python3
"""Compare matched CPU and EP output changes before/after a QDQ ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from run_acc import file_sha256, measure_float_output


def read_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("output_arrays") or not report.get("inputs_sha256"):
        raise ValueError(f"{path}: report must contain --inputs and --output-arrays")
    if report.get("cpu_fallback") or not report.get("ep_context_nodes"):
        raise ValueError(f"{path}: EPContext and disabled CPU fallback are required")
    return report


def validate_external_data(baseline: dict[str, Any], variant: dict[str, Any]) -> None:
    original = baseline["external_data_sha256"]
    changed = variant["external_data_sha256"]
    if any(original.get(location) != digest for location, digest in changed.items()):
        raise ValueError("reports differ in external_data_sha256 for shared or new files")
    removed_locations = original.keys() - changed.keys()
    if not removed_locations:
        return

    models = []
    for report in (baseline, variant):
        path = Path(report["model"])
        if not path.is_file() or file_sha256(path) != report["model_sha256"]:
            raise ValueError(f"model is missing or changed: {path}")
        models.append(onnx.load(path, load_external_data=False))
    source, ablated = models
    retained = {initializer.name for initializer in ablated.graph.initializer}
    uses: dict[str, list[tuple[str, int]]] = {}
    for node in source.graph.node:
        for index, name in enumerate(node.input):
            uses.setdefault(name, []).append((node.op_type, index))
    checked_locations = set()
    for initializer in source.graph.initializer:
        location = next((entry.value for entry in initializer.external_data if entry.key == "location"), None)
        if location not in removed_locations:
            continue
        checked_locations.add(location)
        references = uses.get(initializer.name, [])
        if initializer.name in retained or not references or any(
            op not in ("QuantizeLinear", "DequantizeLinear") or index not in (1, 2)
            for op, index in references
        ):
            raise ValueError(f"external_data_sha256: removed file has retained or non-QDQ data: {location}")
    if checked_locations != removed_locations:
        raise ValueError(f"external_data_sha256: unaccounted removed files: {removed_locations - checked_locations}")


def compare_reports(baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    if baseline.get("cpu_optimization", "all") != variant.get("cpu_optimization", "all"):
        raise ValueError("reports differ in CPU graph optimization level")
    for key in ("provider", "provider_options", "inputs_sha256", "valid_mask", "output_names"):
        if baseline[key] != variant[key]:
            raise ValueError(f"reports differ in {key}: {baseline[key]!r} vs {variant[key]!r}")
    for key in ("ort_version", "provider_library_sha256", "device"):
        if baseline.get(key) != variant.get(key):
            raise ValueError(f"reports differ in {key}: {baseline.get(key)!r} vs {variant.get(key)!r}")
    validate_external_data(baseline, variant)
    if baseline["ablation_pairs_removed"] >= variant["ablation_pairs_removed"]:
        raise ValueError("variant must bypass more QDQ pairs than the baseline")

    input_path = Path(baseline["inputs"])
    if not input_path.is_file() or file_sha256(input_path) != baseline["inputs_sha256"]:
        raise ValueError(f"input archive is absent or has changed: {input_path}")
    with np.load(input_path, allow_pickle=False) as inputs:
        mask = inputs[baseline["valid_mask"]] if baseline["valid_mask"] else None
    if mask is not None and mask.dtype != np.dtype(np.bool_):
        raise ValueError("valid mask must be bool")

    results: dict[str, Any] = {}
    with np.load(baseline["output_arrays"], allow_pickle=False) as original, np.load(
        variant["output_arrays"], allow_pickle=False
    ) as ablated:
        for index, name in enumerate(baseline["output_names"]):
            original_cpu, variant_cpu = original[f"cpu_{index}"], ablated[f"cpu_{index}"]
            original_ep, variant_ep = original[f"ep_{index}"], ablated[f"ep_{index}"]
            if not (original_cpu.shape == variant_cpu.shape == original_ep.shape == variant_ep.shape):
                raise ValueError(f"{name}: outputs have different shapes")
            if mask is not None:
                if original_cpu.shape[: mask.ndim] != mask.shape or not np.any(mask):
                    raise ValueError(f"{name}: valid mask does not select matching output elements")
                original_cpu, variant_cpu = original_cpu[mask], variant_cpu[mask]
                original_ep, variant_ep = original_ep[mask], variant_ep[mask]
            if not np.issubdtype(original_cpu.dtype, np.inexact):
                raise ValueError(f"{name}: delta analysis requires floating-point outputs")
            cpu_shift = measure_float_output(original_cpu, variant_cpu, 0, 0)
            ep_shift = measure_float_output(original_ep, variant_ep, 0, 0)
            original_gap = measure_float_output(original_cpu, original_ep, 0, 0)
            variant_gap = measure_float_output(variant_cpu, variant_ep, 0, 0)
            results[name] = {
                "cpu_shift_mae": cpu_shift["mean_abs_error"],
                "ep_shift_mae": ep_shift["mean_abs_error"],
                "baseline_ep_cpu_mae": original_gap["mean_abs_error"],
                "variant_ep_cpu_mae": variant_gap["mean_abs_error"],
                "baseline_ep_cpu_cosine": original_gap["cosine_similarity"],
                "variant_ep_cpu_cosine": variant_gap["cosine_similarity"],
            }
    return {
        "provider": baseline["provider"],
        "input_sha256": baseline["inputs_sha256"],
        "pairs_added_to_bypass": variant["ablation_pairs_removed"] - baseline["ablation_pairs_removed"],
        "valid_only": mask is not None,
        "outputs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline run_acc.py --report-json file.")
    parser.add_argument("variant", type=Path, help="Ablated run_acc.py --report-json file.")
    parser.add_argument("--output", type=Path, help="New machine-readable JSON delta report.")
    args = parser.parse_args()
    result = compare_reports(read_report(args.baseline), read_report(args.variant))
    print(f"{result['provider']}: {result['pairs_added_to_bypass']} additional QDQ pairs bypassed")
    for name, metrics in result["outputs"].items():
        print(
            f"{name}: CPU shift MAE={metrics['cpu_shift_mae']:.6g}, "
            f"EP shift MAE={metrics['ep_shift_mae']:.6g}; "
            f"CPU/EP cosine {metrics['baseline_ep_cpu_cosine']:.6g} -> {metrics['variant_ep_cpu_cosine']:.6g}"
        )
    if args.output is not None:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
