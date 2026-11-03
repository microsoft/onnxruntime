import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess

from build_qmoe_node_replay import sha
from run_dense_columns16_synthetic import CASES, FLAGS, workspace


FLAGS += " ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER=1"


UUID = "GPU-68170cb3-c9ea-a588-7558-ae82635d9173"


def run_guard(command, prefix):
    from run_fp16_dot4_tree_probe import GUARD
    for suffix in (".log", ".guard.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    child = subprocess.run(["python3", str(GUARD), "--gpu-index", "4",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command])
    proof = json.loads(prefix.with_suffix(".guard.json").read_text())
    valid = ((proof["status"] == "completed" and child.returncode == 0)
             or (proof["status"] == "failed" and child.returncode == 2))
    if (not valid or proof.get("exit_code") != child.returncode or proof["gpu_uuid"] != UUID):
        raise RuntimeError(f"GPU4 proof child did not complete: {prefix.with_suffix('.log')}")
    return child.returncode, prefix.with_suffix(".log").read_text()

ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns8-spatial-double-buffer/build-v1")
DATA = BUILD.parent / "actual-batch-v1"
FLAG = "ORT_WEBGPU_DENSE_COLUMNS8_SPATIAL_DOUBLE_BUFFER"
ORIGINAL = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/libonnxruntime_providers_webgpu.so")


if __name__ == "__main__":
    output = ROOT / "dense-columns8-spatial-double-buffer-actual-batch-v1-outcome.json"
    execution = ROOT / "dense-columns8-spatial-double-buffer-actual-batch-v1-execution.json"
    if output.exists() or execution.exists() or DATA.exists():
        raise FileExistsError("Existing columns8 batch evidence")
    parent_path = ROOT / "dense-paired-matrix-reset-actual-v1-outcome.json"
    parent = json.loads(parent_path.read_text())
    if parent["status"] != "packed_dense_original_provider_actual_weights_supported":
        raise RuntimeError("Underlying reference fixtures not qualified")
    for path, digest in parent["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed underlying fixtures: {path}")
    build_path = BUILD / "build-manifest.json"
    build = json.loads(build_path.read_text())
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    binary = BUILD / "paired_matrix_compare"
    if sha(provider) != build["provider_sha256"] or sha(ORIGINAL) != build["baseline_provider_sha256"]:
        raise RuntimeError("Changed providers")
    if sha(binary) != build["paired_comparator_sha256"]:
        raise RuntimeError("Changed comparator")
    for relative, digest in build["source_sha256"].items():
        if sha(BUILD / relative) != digest:
            raise RuntimeError(f"Changed candidate build: {relative}")
    DATA.mkdir()
    cases = [(k, n, Path("/home/sayanshaw/repos/qwen38-int4-dense-paired-matrix-reset/actual-fixtures-v1") / f"k{k}-n{n}.onnx")
             for k, n in CASES if n != 1]
    evidence = [Path(__file__), ROOT / "build_dense_columns8_spatial_double_buffer.py",
                ROOT / "run_dense_columns8_double_buffer_actual_batch.py", ROOT / "test_dense_columns8_spatial_double_buffer.py",
                ROOT / "prepare_dense_columns8_spatial_double_buffer_proofs.py",
                ROOT / "run_dense_columns8_actual_batch.py",
                parent_path, build_path, provider, ORIGINAL, binary,
                *[model for _, _, model in cases]]
    records = []
    rows = []
    captures = {}
    held = None
    execution.write_text(json.dumps(dict(status="running", records=records), indent=2) + "\n")
    for phase, library, mode, selected in (
        ("capture", ORIGINAL, "capture", 0), ("default", provider, "compare", 0),
        ("candidate", provider, "compare", 1),
    ):
        commands = []
        prefix = ROOT / f"dense-columns8-spatial-double-buffer-actual-batch-v1-{phase}-gpu4"
        for k, n, model in cases:
            reference = DATA / f"k{k}-n{n}-reference.f16"
            if phase == "capture" and reference.exists():
                raise FileExistsError(reference)
            if phase != "capture" and sha(reference) != captures[str(reference)]:
                raise RuntimeError("Changed independent reference")
            args = ["taskset", "-c", "56-63", workspace(binary), mode, workspace(library), workspace(model),
                    workspace(reference), f"/evidence/webgpu-perf-oct20/{prefix.name}-k{k}-n{n}-profile"]
            commands.append(shlex.join(args))
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                        "--format=csv,noheader"], check=True)
        command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
                   f"export {FLAGS} {FLAG}={selected}; " + "; ".join(commands))
        code, log = run_guard(command, prefix)
        records.append(dict(phase=phase, exit_code=code, prefix=str(prefix)))
        execution.write_text(json.dumps(dict(status="running", records=records), indent=2) + "\n")
        evidence.extend((prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")))
        matches = re.findall(r"^DENSE comparisons=(\d+) mismatches=(\d+) K=(\d+) N=(\d+)$", log, re.M)
        profiles = re.findall(r"^PROFILE /evidence/webgpu-perf-oct20/(.+)$", log, re.M)
        if "FAIL" in log or len(matches) != len(profiles) or not matches:
            raise RuntimeError("Incomplete batch arithmetic/profile")
        if len(matches) != len(cases) and (phase != "candidate" or code != 2):
            raise RuntimeError("Incomplete original/default batch")
        phase_rows = []
        for (count, mismatch, k, n), profile_name, expected in zip(matches, profiles, cases):
            count, mismatch, k, n = map(int, (count, mismatch, k, n))
            if (k, n) != expected[:2] or count != n * 64 or not 0 <= mismatch <= count:
                raise RuntimeError("Wrong batch coverage")
            if phase != "candidate" and mismatch:
                raise RuntimeError("Original/default differs")
            profile = ROOT / profile_name
            events = json.loads(profile.read_text())
            dispatched = [e for e in events if e.get("cat") == "Kernel" and "&MatMulNBits&" in e.get("name", "")]
            chosen = sum("DenseColumns8SpatialDoubleBuffer" in e.get("name", "") for e in dispatched)
            if len(dispatched) != 64 or chosen != (64 if selected and n >= 2560 else 0):
                raise RuntimeError("Missing selected/fallback dispatch proof")
            parent_chosen = sum("DenseColumns8DoubleBuffer" in e.get("name", "") for e in dispatched)
            if parent_chosen != (64 if phase == "default" and n >= 2560 else 0):
                raise RuntimeError("Missing supported-parent/default dispatch proof")
            if any(e.get("dur", 0) <= 0 for e in dispatched):
                raise RuntimeError("Invalid isolated timing")
            reference = DATA / f"k{k}-n{n}-reference.f16"
            if phase == "capture":
                if reference.stat().st_size != n * 16 * 2:
                    raise RuntimeError("Incomplete production reference")
                captures[str(reference)] = sha(reference)
                evidence.append(reference)
            elif sha(reference) != captures[str(reference)]:
                raise RuntimeError("Independent reference changed")
            evidence.append(profile)
            phase_rows.append(dict(k=k, n=n, comparisons=count, mismatches=mismatch,
                                   selected_dispatches=chosen,
                                   instrumented_gpu_median_us=statistics.median(e["dur"] for e in dispatched)))
        rows.append(dict(phase=phase, shapes=phase_rows))
        if code != (2 if any(row["mismatches"] for row in phase_rows) else 0):
            raise RuntimeError("Batch numerical verdict disagrees with exit")
        print(json.dumps(rows[-1]), flush=True)
        if code == 2:
            held = "held_columns8_spatial_double_buffer_original_kernel_bitwise_mismatches"
            break
    status = held or "columns8_spatial_double_buffer_batched_actual_and_default_supported"
    execution.write_text(json.dumps(dict(status=status, records=records), indent=2) + "\n")
    evidence.append(execution)
    report = dict(status=status, phases=rows, batched_container_phases=True,
                  separate_process_per_fixture=True, model_eligible=False, default_enabled=False,
                  actual_model_weights_used=True, natural_model_timing_performed=False,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=status)), flush=True)
