import json
from pathlib import Path
import re
import shlex
import subprocess

import onnx
from onnx import TensorProto, helper

from build_argmax_parallel_v3 import FLAG
from build_qmoe_node_replay import sha
from run_dense_columns8_spatial_double_buffer_batch import UUID
from run_dense_columns16_synthetic import workspace
from run_fp16_dot4_tree_probe import GUARD


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-argmax-parallel/build-v3")
ORIGINAL = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/libonnxruntime_providers_webgpu.so")
DATA = BUILD.parent / "fixtures-v3"


def guarded(command, prefix):
    for suffix in (".log", ".guard.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader"], check=True)
    child = subprocess.run(["python3", str(GUARD), "--gpu-index", "4", "--idle-seconds", "3",
                            "--wait-seconds", "15", "--run-seconds", "900", "--output-prefix", str(prefix),
                            "--", "bash", "-lc", command])
    proof = json.loads(prefix.with_suffix(".guard.json").read_text())
    if (proof.get("gpu_uuid") != UUID or proof.get("exit_code") != child.returncode or
            proof["status"] not in ("completed", "failed") or child.returncode not in (0, 1, 2)):
        raise RuntimeError(f"GPU4 guard did not prove a completed child: {prefix}")
    return child.returncode, prefix.with_suffix(".log").read_text()


def profile_counts(path):
    counts = {}
    for event in json.loads(path.read_text()):
        name = event.get("name", "")
        if event.get("cat") == "Kernel" and "&ArgMax&" in name:
            program = next((p for p in ("ArgMaxParallelPartial", "ArgMaxParallelFinish") if p in name), "ArgMax")
            counts[program] = counts.get(program, 0) + 1
    return counts


def host_path(path):
    for prefix, root in (("/workspace/", Path("/home/sayanshaw/repos")),
                         ("/evidence/", ROOT.parent)):
        if path.startswith(prefix):
            return root / path[len(prefix):]
    raise RuntimeError(f"Unexpected container path: {path}")


def main():
    output = ROOT / "argmax-parallel-v3-native-outcome.json"
    if output.exists() or DATA.exists():
        raise FileExistsError("Existing native argmax evidence")
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    binary = BUILD / "argmax_compare"
    for path, digest in manifest["source_sha256"].items():
        host = host_path(path)
        if sha(host) != digest:
            raise RuntimeError(f"Changed argmax build evidence: {host}")
    if sha(provider) != manifest["provider_sha256"] or sha(ORIGINAL) != manifest["baseline_provider_sha256"]:
        raise RuntimeError("Changed provider")
    if sha(binary) != manifest["comparator_sha256"]:
        raise RuntimeError("Changed comparator")
    DATA.mkdir()
    fixtures = []
    for dtype, label in ((TensorProto.FLOAT, "f32"), (TensorProto.FLOAT16, "f16")):
        for rows, n in ((1, 17), (1, 32), (3, 2049), (1, 248320), (3, 248320)):
            path = DATA / f"{label}-rows{rows}-n{n}.onnx"
            graph = helper.make_graph(
                [helper.make_node("ArgMax", ["x"], ["y"], axis=1, keepdims=0, select_last_index=0)],
                "argmax-exact", [helper.make_tensor_value_info("x", dtype, [rows, n])],
                [helper.make_tensor_value_info("y", TensorProto.INT64, [rows])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
            onnx.checker.check_model(model)
            onnx.save(model, path)
            fixtures.append((label, rows, n, path))
    evidence = [Path(__file__), ROOT / "build_argmax_parallel_v3.py", ROOT / "argmax_parallel_compare_v3.cpp",
                ROOT / "test_argmax_parallel.py", manifest_path, provider, ORIGINAL, binary, GUARD,
                *[row[-1] for row in fixtures]]
    phases = []
    head = f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
    status = "argmax_parallel_native_exact_and_dispatch_supported"
    for phase, library, mode in (("original", ORIGINAL, 0), ("default", provider, 0), ("selected", provider, 1)):
        prefix = ROOT / f"argmax-parallel-v3-{phase}-gpu4"
        commands = []
        for label, rows, n, path in fixtures:
            profile_prefix = f"/evidence/webgpu-perf-oct20/{prefix.name}-{label}-rows{rows}-n{n}"
            commands.append(shlex.join(["taskset", "-c", "48", workspace(binary), "compare",
                                        workspace(library), workspace(path), profile_prefix]))
        code, log = guarded(head + f"export {FLAG}={mode}; " + "; ".join(commands), prefix)
        evidence.extend([prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")])
        if code != 0 or "FAIL" in log:
            status = "held_native_argmax_failure"
            phases.append(dict(phase=phase, exit_code=code, log=str(prefix.with_suffix(".log"))))
            break
        counts = re.findall(r"^ARGMAX rows=(\d+) N=(\d+) comparisons=(\d+) mismatches=0$", log, re.M)
        profiles = re.findall(r"^PROFILE (.+)$", log, re.M)
        if len(counts) != len(fixtures) or len(profiles) != len(fixtures):
            raise RuntimeError("Incomplete native comparison/profile coverage")
        shapes = []
        for (label, rows, n, _), observed, profile in zip(fixtures, counts, profiles, strict=True):
            if tuple(map(int, observed)) != (rows, n, rows * 64):
                raise RuntimeError("Incorrect native comparison shape/count")
            profile_path = host_path(profile)
            dispatches = profile_counts(profile_path)
            selected = mode == 1 and n >= 32
            if (dispatches.get("ArgMaxParallelPartial", 0) != (64 if selected else 0) or
                    dispatches.get("ArgMaxParallelFinish", 0) != (64 if selected else 0) or
                    dispatches.get("ArgMax", 0) != (0 if selected else 64)):
                raise RuntimeError(f"Incorrect selected/default/fallback dispatches: {dispatches}")
            evidence.append(profile_path)
            shapes.append(dict(dtype=label, rows=rows, n=n, comparisons=rows * 64,
                               mismatches=0, dispatches=dispatches))
        phases.append(dict(phase=phase, shapes=shapes))
    result = dict(status=status, phases=phases, gpu_uuid=UUID, model_eligible=False, default_enabled=False,
                  natural_model_tps_measured=False, independent_cpu_argmax_reference=True,
                  nonfinite_inputs_signed_zero_subnormals_and_ties_included=True,
                  provider_sha256=manifest["provider_sha256"],
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=status, phases=phases)), flush=True)


if __name__ == "__main__":
    main()
