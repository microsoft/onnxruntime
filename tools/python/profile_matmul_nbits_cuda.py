# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Capture one CUDA MatMulNBits case or profile its warmed-up GPU-resident execution."""

import argparse
import json
import os
import platform
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

if __package__:
    from .run_matmul_nbits_experiments import (
        CaseError,
        artifact_path,
        load_activation,
        load_manifest,
        validate_report,
    )
else:
    from run_matmul_nbits_experiments import (
        CaseError,
        artifact_path,
        load_activation,
        load_manifest,
        validate_report,
    )


def run(args):
    import onnxruntime as ort  # noqa: PLC0415

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise ValueError("CUDAExecutionProvider unavailable: install a CUDA wheel built from this branch")
    if args.device < 0 or args.warmup < 1 or args.repeats < 1:
        raise ValueError("device must be nonnegative; warmup and repeats must be positive")
    if args.mode == "capture" and args.nvtx:
        raise ValueError("--nvtx applies only to profile mode")
    annotate = None
    if args.nvtx:
        import nvtx  # noqa: PLC0415

        annotate = nvtx.annotate
    manifest = args.manifest.resolve()
    matches = [(record, case) for record, case in load_manifest(manifest) if record["case_id"] == args.case_id]
    if len(matches) != 1:
        raise ValueError("case-id must select exactly one manifest case")
    record, case = matches[0]
    activation = load_activation(
        artifact_path(manifest.parent, record["model"]), artifact_path(manifest.parent, record["inputs"]), case
    )
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap,driver_version", "--format=csv"], text=True
    )
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "complete": False,
        "mode": args.mode,
        "case_id": args.case_id,
        "settings": record["settings"],
        "runtime_file": ort.__file__,
        "runtime_version": ort.__version__,
        "runtime_build": ort.get_build_info(),
        "python": sys.executable,
        "platform": platform.platform(),
        "gpu_inventory": gpu_info,
        "cuda_device_ordinal": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "scope": "GPU-resident IO; separate count capture and profiling; not measured FLOP accuracy",
        "nvtx_range": "matmul_nbits_measure" if args.nvtx else None,
    }

    def save():
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    save()

    def session(reporting):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if reporting:
            options.add_session_config_entry("session.compute_estimate_report_path", str(args.output / "estimate"))
        value = ort.InferenceSession(
            str(manifest.parent / record["model"]),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"device_id": args.device, "enable_cuda_graph": "0"})],
        )
        value.disable_fallback()
        if value.get_providers()[0] != "CUDAExecutionProvider":
            raise ValueError("CUDA provider initialization fell back to another provider")
        return value

    def bind(value):
        tensor = ort.OrtValue.ortvalue_from_numpy(activation, "cuda", args.device)
        binding = value.io_binding()
        binding.bind_ortvalue_input("A", tensor)
        binding.bind_output("Y", "cuda", args.device)
        binding.synchronize_inputs()
        return tensor, binding

    baseline = session(False)
    _baseline_tensor, baseline_io = bind(baseline)
    for _ in range(args.warmup):
        baseline.run_with_iobinding(baseline_io)
    baseline_io.synchronize_outputs()
    reference = baseline_io.copy_outputs_to_cpu()[0]
    if (
        not np.isfinite(reference).all()
        or reference.shape != (*case.a_shape[:-1], case.n)
        or reference.dtype != np.dtype(case.dtype)
    ):
        raise ValueError("Invalid baseline output")
    if args.mode == "capture":
        captured = session(True)
        _captured_tensor, captured_io = bind(captured)
        if list(args.output.glob("estimate_*.json")):
            raise ValueError("Initialization unexpectedly exported a report")
        captured.run_with_iobinding(captured_io)
        captured_io.synchronize_outputs()
        observed = captured_io.copy_outputs_to_cpu()[0]
        if observed.dtype != reference.dtype or not np.array_equal(reference, observed):
            raise ValueError("Reporting-on/off outputs differ or contain nonfinite values")
        reports = list(args.output.glob("estimate_*.json"))
        if len(reports) != 1:
            raise ValueError(f"Expected one report, found {len(reports)}; check wheel provenance")
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        validate_report(report, case, provider="CUDAExecutionProvider")
        metadata["invocation"] = report["invocations"][0]
        np.savez(args.output / "outputs.npz", reporting_disabled=reference, reporting_enabled=observed)
    else:
        with annotate("matmul_nbits_measure") if annotate else nullcontext():
            for _ in range(args.repeats):
                baseline.run_with_iobinding(baseline_io)
            baseline_io.synchronize_outputs()
        metadata["note"] = "Route labels require a separate capture run; NVTX range excludes initialization/warmup."
    metadata["complete"] = True
    save()
    print(f"Completed {args.mode}: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths rejected")
    parser.add_argument("--mode", choices=("capture", "profile"), default="capture")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ordinal; respect CUDA_VISIBLE_DEVICES")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--nvtx", action="store_true", help="Require nvtx package and mark only the measured region")
    args = parser.parse_args()
    try:
        run(args)
    except (CaseError, OSError, ValueError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"CUDA collection failed: {error}\n")


if __name__ == "__main__":
    main()
