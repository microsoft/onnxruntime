# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root.
# --------------------------------------------------------------------------
"""Compare CPU/CUDA outputs and reject behavior-changing intermediate captures."""

import argparse
import hashlib
import io
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

import onnxruntime as ort


def load_inputs(path):
    data = path.read_bytes()
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Duplicate input archive names")
        feeds = {name: archive[name] for name in archive.files}
    return feeds, hashlib.sha256(data).hexdigest()


def compare_outputs(reference, candidate, atol=1e-4, rtol=1e-4):
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("Output shape/dtype mismatch")
    if reference.dtype.kind not in "biuf":
        raise ValueError("Only numeric/bool tensors are supported")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite output")
    max_absolute = None
    if reference.dtype.kind == "f":
        with np.errstate(over="raise", invalid="raise"):
            expected = reference.astype(np.float64)
            difference = np.abs(candidate.astype(np.float64) - expected)
            mismatch = difference > atol + rtol * np.abs(expected)
        max_absolute = float(np.max(difference, initial=0))
    else:
        mismatch = candidate != reference
    coordinates = [
        list(np.unravel_index(int(index), reference.shape)) for index in np.flatnonzero(mismatch.reshape(-1))[:8]
    ]
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "elements": int(reference.size),
        "bitwise_equal": reference.tobytes(order="C") == candidate.tobytes(order="C"),
        "failing_elements": int(np.count_nonzero(mismatch)),
        "max_absolute_difference": max_absolute,
        "first_failing_coordinates": [[int(value) for value in coordinate] for coordinate in coordinates],
    }


def compare_captures(cpu, cuda, captured_cpu=None, captured_cuda=None, atol=1e-4, rtol=1e-4):
    if not cpu or not cuda:
        raise ValueError("Original model must have outputs")
    if cpu.keys() != cuda.keys():
        raise ValueError("CPU/CUDA output names differ")
    baseline = {name: compare_outputs(value, cuda[name], atol, rtol) for name, value in cpu.items()}
    result = {"baseline": baseline, "instrumentation_preserved": None, "captures": None}
    if (captured_cpu is None) != (captured_cuda is None):
        raise ValueError("Both instrumented provider runs are required")
    if captured_cpu is not None:
        if captured_cpu.keys() != captured_cuda.keys() or not cpu.keys() <= captured_cpu.keys():
            raise ValueError("Instrumented outputs must include all original outputs, with matching provider names")
        checks = {
            provider: {
                name: compare_outputs(value, captured[name], atol, rtol)["bitwise_equal"]
                for name, value in original.items()
            }
            for provider, original, captured in (("cpu", cpu, captured_cpu), ("cuda", cuda, captured_cuda))
        }
        preserved = all(all(outputs.values()) for outputs in checks.values())
        result.update(instrumentation_preserved=preserved, preservation_checks=checks)
        if preserved:
            result["captures"] = {
                name: compare_outputs(value, captured_cuda[name], atol, rtol) for name, value in captured_cpu.items()
            }
    return result


def run_model(model, feeds, provider, directory):
    directory.mkdir()
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"Requested execution provider is unavailable: {provider}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 16
    options.inter_op_num_threads = 1
    options.enable_profiling = True
    options.profile_file_prefix = str(directory / "profile")
    providers = [provider]
    if provider == "CUDAExecutionProvider":
        ort.preload_dlls(directory="")
        providers = [(provider, {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    effective = session.get_provider_options()
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"Requested provider was not initialized: {provider}")
    if provider == "CUDAExecutionProvider" and effective[provider].get("use_tf32") != "0":
        raise RuntimeError("TF32 was not disabled")
    session.disable_fallback()
    if set(feeds) != {value.name for value in session.get_inputs()}:
        raise ValueError("Input archive must match the model's input names exactly")
    names = [value.name for value in session.get_outputs()]
    values = session.run(None, feeds)
    profile_path = Path(session.end_profiling())
    del session
    events = [
        event
        for event in json.loads(profile_path.read_bytes())
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    ]
    placement = Counter(event["args"]["provider"] for event in events)
    if provider == "CUDAExecutionProvider" and not placement["CUDAExecutionProvider"]:
        raise RuntimeError("Profile contains no CUDA kernel execution")
    outputs = dict(zip(names, values, strict=True))
    artifacts = {}
    for index, (name, array) in enumerate(outputs.items()):
        path = directory / f"output-{index:04d}.npy"
        with path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        artifacts[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return outputs, {
        "graph_sha256_excludes_external_data": hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        "effective_provider_options": effective,
        "profile_path": str(profile_path),
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "kernel_counts": dict(placement),
        "operations": dict(Counter(event["args"]["op_name"] for event in events)),
        "cpu_nodes_require_review": [
            {"name": event["name"], "args": event["args"]}
            for event in events
            if event["args"]["provider"] == "CPUExecutionProvider"
        ],
        "outputs": artifacts,
    }


def exit_code(comparison):
    if comparison["instrumentation_preserved"] is False:
        return 3
    summaries = [comparison["baseline"], comparison["captures"] or {}]
    return 2 if any(value["failing_elements"] for outputs in summaries for value in outputs.values()) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="NPZ with exact model input names; no pickle")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--instrumented-model", type=Path, help="Same model with additional outputs")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    compare_outputs(np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), args.atol, args.rtol)
    feeds, inputs_digest = load_inputs(args.inputs)
    args.output_dir.mkdir()
    cpu, cpu_record = run_model(args.model, feeds, "CPUExecutionProvider", args.output_dir / "cpu")
    cuda, cuda_record = run_model(args.model, feeds, "CUDAExecutionProvider", args.output_dir / "cuda")
    captured_cpu = captured_cuda = None
    runs = {"cpu": cpu_record, "cuda": cuda_record}
    if args.instrumented_model:
        captured_cpu, runs["captured_cpu"] = run_model(
            args.instrumented_model, feeds, "CPUExecutionProvider", args.output_dir / "captured-cpu"
        )
        captured_cuda, runs["captured_cuda"] = run_model(
            args.instrumented_model, feeds, "CUDAExecutionProvider", args.output_dir / "captured-cuda"
        )
    comparison = compare_captures(cpu, cuda, captured_cpu, captured_cuda, args.atol, args.rtol)
    report = {
        "diagnostic_only": True,
        "cpu_is_not_an_independent_source_oracle": True,
        "kernel_defect_established": False,
        "ort_version": ort.__version__,
        "ort_build": ort.get_build_info(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "optimization_level": "ORT_ENABLE_ALL",
        "inputs_sha256": inputs_digest,
        "atol": args.atol,
        "rtol": args.rtol,
        "runs": runs,
        "comparison": comparison,
    }
    destination = args.output_dir / "comparison.json"
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(destination)
    return exit_code(comparison)


if __name__ == "__main__":
    raise SystemExit(main())
