import json
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from build_dense_columns16 import FLAG
from build_qmoe_node_replay import sha
from run_fp16_dot4_tree_probe import UUID, workspace


ROOT = Path(__file__).parent
DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-dense-columns16")
BUILD = DIRECTORY / "build-v1"
DATA = DIRECTORY / "actual-fixtures-v1"
ORIGINAL = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/libonnxruntime_providers_webgpu.so")
CASES = ((640, 2560), (2560, 2560), (2560, 7168), (6144, 2560),
         (10240, 320), (2560, 16384), (2560, 6144), (2560, 10240))
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


MODEL_DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
MODEL_GRAPH = MODEL_DIRECTORY / "model.onnx"
MODEL_CONFIG = MODEL_DIRECTORY / "genai_config.json"


def fixture(k, n):
    import hashlib
    from onnx.external_data_helper import load_external_data_for_tensor
    if sha(MODEL_GRAPH) != "8037bd0a14a3ae5307cd2d626e90d3be26a6a7b8893062246ee23487599400f9":
        raise RuntimeError("Changed frozen model graph")
    if sha(MODEL_CONFIG) != "8b878a2c79a0763448f02a5105aa98a3e32f43ded848f93a1df9a071607e16b6":
        raise RuntimeError("Changed frozen model config")
    model = onnx.load(MODEL_GRAPH, load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    selected = None
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if (attrs.get("K"), attrs.get("N")) == (k, n):
            if attrs.get("bits") != 4 or attrs.get("block_size") != 32 or len(node.input) != 3:
                raise RuntimeError("Unexpected actual-weight layout")
            selected = node
            break
    if selected is None:
        raise RuntimeError(f"No actual model shape: {(k, n)}")
    tensors = []
    provenance = dict(k=k, n=n, node=selected.name, tensors=[])
    for source_name, target in zip(selected.input[1:3], ("b", "scales")):
        tensor = initializers[source_name]
        external = {item.key: item.value for item in tensor.external_data}
        if tensor.data_location == TensorProto.EXTERNAL:
            load_external_data_for_tensor(tensor, str(MODEL_DIRECTORY))
        array = numpy_helper.to_array(tensor, base_dir=str(MODEL_DIRECTORY))
        expected = (n, k // 32, 16) if target == "b" else (n, k // 32)
        dtype = np.uint8 if target == "b" else np.float16
        if array.shape != expected or array.dtype != dtype or not np.isfinite(array).all():
            raise RuntimeError("Invalid actual-weight tensor")
        provenance["tensors"].append(dict(name=source_name, external=external,
            payload_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
            shape=list(array.shape), dtype=str(array.dtype)))
        tensors.append(numpy_helper.from_array(array, target))
    path = DATA / f"k{k}-n{n}.onnx"
    if path.exists():
        raise FileExistsError(path)
    graph = helper.make_graph(
        [helper.make_node("MatMulNBits", ["a", "b", "scales"], ["y"],
            name="actual_dense", domain="com.microsoft",
            K=k, N=n, bits=4, block_size=32, accuracy_level=4)],
        "dense-paired-matrix-actual",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT16, [1, k])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, n])], tensors)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    provenance["fixture_sha256"] = sha(path)
    record = DATA / f"k{k}-n{n}-actual-tensors.json"
    record.write_text(json.dumps(provenance, indent=2) + "\n")
    return path


if __name__ == "__main__":
    outcome = ROOT / "dense-columns16-actual-v1-outcome.json"
    execution = ROOT / "dense-columns16-actual-v1-execution.json"
    if execution.exists() or outcome.exists() or DATA.exists():
        raise FileExistsError("Existing packed dense evidence")
    synthetic_path = ROOT / "dense-columns16-v1-outcome.json"
    synthetic = json.loads(synthetic_path.read_text())
    if synthetic["status"] != "dense_columns16_original_provider_synthetic_shapes_supported":
        raise RuntimeError("Synthetic original-provider gate not supported")
    for path, digest in synthetic["source_and_evidence_sha256"].items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed synthetic evidence: {path}")
    prerequisite = ROOT / "fp16-dot4-dynamic-sweep-v1-outcome.json"
    primitive = json.loads(prerequisite.read_text())
    if primitive["status"] != "dynamic_finite_half_accumulator_sweep_supported":
        raise RuntimeError("Broad arithmetic prerequisite not supported")
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
    evidence = [synthetic_path, MODEL_GRAPH, MODEL_CONFIG, Path(__file__), ROOT / "build_dense_columns16.py",
                ROOT / "test_dense_columns16.py", ROOT / "dense_paired_matrix_compare.cpp",
                prerequisite, build_path, ORIGINAL, provider, binary]
    execution.write_text(json.dumps(dict(status="running", records=records), indent=2) + "\n")
    held = None
    for k, n in CASES:
        model = fixture(k, n)
        reference = DATA / f"k{k}-n{n}-reference.f16"
        evidence.extend((model, DATA / f"k{k}-n{n}-actual-tensors.json"))
        reference_digest = None
        phases = []
        for mode, selected, library in (("capture", 0, ORIGINAL), ("default", 0, provider), ("compare", 1, provider)):
            subprocess.run(["nvidia-smi",
                "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader"], check=True)
            prefix = ROOT / f"dense-columns16-actual-v1-k{k}-n{n}-{mode}-gpu7"
            args = [workspace(binary), "compare" if mode == "default" else mode, workspace(library), workspace(model),
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
            selected_events = [event for event in events if "DenseColumns16" in event.get("name", "")]
            if mode == "compare" and n != 1 and len(selected_events) != 64:
                raise RuntimeError(f"Selected kernel dispatch coverage={len(selected_events)}, expected64: {profile}")
            if mode in ("capture", "default") or n == 1:
                if selected_events:
                    raise RuntimeError("Unexpected selected kernel in original/fallback")
            evidence.append(profile)
            if mode == "capture":
                if reference.stat().st_size != n * 16 * 2:
                    raise RuntimeError("Incomplete production reference")
                reference_digest = sha(reference)
                evidence.append(reference)
            elif sha(reference) != reference_digest:
                raise RuntimeError("Production reference changed")
            if mode == "default" and mismatches:
                raise RuntimeError("Compiled default-off provider differs from original")
            phases.append(dict(mode=mode, comparisons=count, mismatches=mismatches,
                               selected_dispatches=len(selected_events)))
        results.append(dict(k=k, n=n, phases=phases))
        print(json.dumps(results[-1]), flush=True)
        if phases[-1]["mismatches"]:
            held = "held_original_production_kernel_bitwise_mismatches"
            break
    status = held or "dense_columns16_original_provider_actual_weights_supported"
    execution.write_text(json.dumps(dict(status=status, records=records), indent=2) + "\n")
    evidence.append(execution)
    report = dict(status=status, shapes=results, default_enabled=False, model_eligible=False,
                  actual_model_weights_used=True, model_timing_performed=False,
                  invalid_selector_tested=False,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=status, shapes=results)), flush=True)
