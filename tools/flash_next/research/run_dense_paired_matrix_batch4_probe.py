import json
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from build_dense_paired_matrix_batch4 import FLAG
from build_qmoe_node_replay import sha
from run_fp16_dot4_tree_probe import UUID, workspace


ROOT = Path(__file__).parent
DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-dense-paired-matrix-batch4")
BUILD = DIRECTORY / "build-v1"
DATA = DIRECTORY / "fixtures-v1"
ORIGINAL = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/libonnxruntime_providers_webgpu.so")
CASES = ((640, 2560),)
FLAGS = ("ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_TILE4=1 "
         "ORT_WEBGPU_DENSE_BLOCK32=1 ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1")


def run_guard(command, prefix):
    from run_fp16_dot4_tree_probe import GUARD
    for suffix in (".log", ".guard.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    child = subprocess.run(["python3", str(GUARD), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command])
    proof = json.loads(prefix.with_suffix(".guard.json").read_text())
    valid = ((proof["status"] == "completed" and child.returncode == 0) or
             (proof["status"] == "failed" and child.returncode == 2))
    if not valid or proof.get("exit_code") != child.returncode or proof["gpu_uuid"] != UUID:
        raise RuntimeError(f"GPU guard did not prove completed child: {prefix.with_suffix('.guard.json')}")
    return child.returncode, prefix.with_suffix(".log").read_text()


def fixture(k, n):
    path = DATA / f"k{k}-n{n}.onnx"
    if path.exists():
        raise FileExistsError(path)
    blocks = k // 32
    codes = ((np.arange(n * blocks * 16, dtype=np.uint32) * 73 + 19) % 256).astype(np.uint8)
    scales = (0.001 * (1 + np.arange(n * blocks, dtype=np.uint32) % 31)).astype(np.float16)
    graph = helper.make_graph(
        [helper.make_node("MatMulNBits", ["a", "b", "scales"], ["y"],
                          name="dense", domain="com.microsoft",
                          K=k, N=n, bits=4, block_size=32, accuracy_level=0)],
        "dense-paired-matrix",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT16, [1, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, n])],
        [numpy_helper.from_array(codes.reshape(n, blocks, 16), "b"),
         numpy_helper.from_array(scales.reshape(n, blocks), "scales")])
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    outcome = ROOT / "dense-paired-matrix-batch4-probe-v1-outcome.json"
    execution = ROOT / "dense-paired-matrix-batch4-probe-v1-execution.json"
    if execution.exists() or outcome.exists() or DATA.exists():
        raise FileExistsError("Existing packed dense evidence")
    prerequisite = ROOT / "fp16-dot4-dynamic-sweep-v1-outcome.json"
    primitive = json.loads(prerequisite.read_text())
    if primitive["status"] != "dynamic_finite_half_accumulator_sweep_supported":
        raise RuntimeError("Broad arithmetic prerequisite not supported")
    limits_path = ROOT / "native-matrix-limit-probe-v1-outcome.json"
    limits = json.loads(limits_path.read_text())
    for path, digest in limits["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed device-limit proof: {path}")
    if (limits["status"] != "actual_device_limits_observed"
            or not limits["paired_matrix_configuration"]
            or limits["workgroup_storage_bytes"] < 46080
            or limits["workgroup_size_x"] < 512
            or limits["workgroup_invocations"] < 512):
        raise RuntimeError("Observed device limits do not admit the new four-slot design")
    build_path = BUILD / "build-manifest.json"
    build = json.loads(build_path.read_text())
    if not build["all_152_original_source_objects_byte_identical"] or not build["baseline_byte_identical"]:
        raise RuntimeError("Unproved original build")
    for relative, digest in build["source_sha256"].items():
        if sha(BUILD / relative) != digest:
            raise RuntimeError(f"Changed source/SDK: {relative}")
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    binary = BUILD / "paired_matrix_compare"
    if sha(ORIGINAL) != build["baseline_provider_sha256"] or sha(provider) != build["provider_sha256"]:
        raise RuntimeError("Changed providers")
    if sha(binary) != build["paired_comparator_sha256"]:
        raise RuntimeError("Changed comparator")
    DATA.mkdir()
    records = []
    results = []
    evidence = [Path(__file__), ROOT / "build_dense_paired_matrix_batch4.py",
                ROOT / "test_dense_paired_matrix_batch4.py", ROOT / "dense_paired_matrix_compare.cpp",
                ROOT / "prepare_dense_paired_matrix_batch4_probe.py",
                prerequisite, limits_path, build_path, ORIGINAL, provider, binary]
    execution.write_text(json.dumps(dict(status="running", records=records), indent=2) + "\n")
    held = None
    for k, n in CASES:
        model = fixture(k, n)
        reference = DATA / f"k{k}-n{n}-reference.f16"
        evidence.append(model)
        reference_digest = None
        phases = []
        for mode, selected, library in (("capture", 0, ORIGINAL), ("compare", 1, provider)):
            subprocess.run(["nvidia-smi",
                "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader"], check=True)
            prefix = ROOT / f"dense-paired-matrix-batch4-probe-v1-k{k}-n{n}-{mode}-gpu7"
            args = [workspace(binary), mode, workspace(library), workspace(model),
                    workspace(reference), "/evidence/webgpu-perf-oct20/" + prefix.name + "-profile"]
            command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
                       f"export {FLAGS} {FLAG}={selected}; exec " + shlex.join(args))
            code, log = run_guard(command, prefix)
            records.append(dict(k=k, n=n, mode=mode, exit_code=code, prefix=str(prefix)))
            execution.write_text(json.dumps(dict(status="running", records=records), indent=2) + "\n")
            evidence.extend((prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")))
            match = re.search(r"^DENSE comparisons=(\d+) mismatches=(\d+) K=(\d+) N=(\d+)$", log, re.M)
            if code not in (0, 2) or "FAIL" in log or not match:
                raise RuntimeError(f"Full-kernel {mode} failed; see {prefix.with_suffix('.log')}")
            count, mismatches, observed_k, observed_n = map(int, match.groups())
            if count != n * 64 or (observed_k, observed_n) != (k, n) or not 0 <= mismatches <= count:
                raise RuntimeError("Incorrect full-kernel comparison coverage")
            if code != (2 if mismatches else 0) or (mode == "capture" and mismatches):
                raise RuntimeError("Full-kernel exit disagrees with arithmetic verdict")
            profile_match = re.search(r"^PROFILE /evidence/webgpu-perf-oct20/(.+)$", log, re.M)
            if not profile_match:
                raise RuntimeError("Missing profiling evidence")
            profile = ROOT / profile_match[1]
            events = json.loads(profile.read_text())
            native_events = [event for event in events if "DensePairedMatrixBatch4" in event.get("name", "")]
            if mode == "compare" and n != 1 and len(native_events) != 64:
                raise RuntimeError(f"Native matrix dispatch coverage={len(native_events)}, expected64: {profile}")
            if mode == "capture" or n == 1:
                if native_events:
                    raise RuntimeError("Unexpected native matrix in original/fallback")
            evidence.append(profile)
            if mode == "capture":
                if reference.stat().st_size != n * 16 * 2:
                    raise RuntimeError("Incomplete production reference")
                reference_digest = sha(reference)
                evidence.append(reference)
            elif sha(reference) != reference_digest:
                raise RuntimeError("Production reference changed")
            phases.append(dict(mode=mode, comparisons=count, mismatches=mismatches,
                               native_dispatches=len(native_events)))
        results.append(dict(k=k, n=n, phases=phases))
        print(json.dumps(results[-1]), flush=True)
        if phases[1]["mismatches"]:
            held = "held_original_production_kernel_bitwise_mismatches"
            break
    status = held or "packed_dense_original_provider_synthetic_shapes_supported"
    execution.write_text(json.dumps(dict(status=status, records=records), indent=2) + "\n")
    evidence.append(execution)
    report = dict(status=status, shapes=results, default_enabled=False, model_eligible=False,
                  actual_model_weights_used=False, model_timing_performed=False,
                  invalid_selector_tested=False,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=status, shapes=results)), flush=True)
