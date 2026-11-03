import json
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper

from build_qmoe_node_replay import sha
from run_argmax_parallel_bitwise import ROOT, UUID, guarded, host_path, profile_counts, workspace


BUILD = Path("/home/sayanshaw/repos/qwen38-int4-gpu-greedy-argmax/build-v4")
DATA = BUILD.parent / "mask-fixtures-v4"


def main():
    output = ROOT / "gpu-greedy-argmax-v4-mask-native-outcome.json"
    if output.exists() or DATA.exists():
        raise FileExistsError("Existing GPU min-length evidence")
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    if sha(provider) != manifest["provider_sha256"]:
        raise RuntimeError("Changed GPU greedy provider")
    for path, digest in manifest["source_sha256"].items():
        if sha(host_path(path)) != digest:
            raise RuntimeError(f"Changed GPU greedy build: {path}")
    cpp, binary = ROOT / "gpu_greedy_mask_compare.cpp", BUILD / "mask_compare"
    subprocess.run(["sudo", "-n", "docker", "-H", "unix:///home/sayanshaw/docker-qwen38.sock",
        "run", "--rm", "--network", "none", "--user", "664503:664503", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "-v", "/home/sayanshaw/repos:/workspace",
        "-v", str(ROOT.parent) + ":/evidence",
        "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel", "taskset", "-c", "56-63",
        "c++", "-O3", "-std=c++20", "/evidence/webgpu-perf-oct20/" + cpp.name,
        "-I/workspace/onnxruntime-webgpu-h200-perf/include/onnxruntime/core/session",
        "-L/workspace/qwen38-fp8-webgpu-runtime/sdk/lib",
        "-Wl,-rpath,/workspace/qwen38-fp8-webgpu-runtime/sdk/lib", "-lonnxruntime",
        "-o", workspace(binary)], check=True)
    DATA.mkdir()
    evidence = [Path(__file__), ROOT / "prepare_gpu_greedy_argmax_v4.py", cpp, binary,
                manifest_path, provider, BUILD / "libonnxruntime-genai.so"]
    fixtures = []
    for dtype, label in ((TensorProto.FLOAT, "f32"), (TensorProto.FLOAT16, "f16")):
        for rows, n in ((1, 32), (3, 2049), (1, 248320), (3, 248320)):
            path = DATA / f"{label}-rows{rows}-n{n}.onnx"
            eos = {0, min(127, n - 1), min(2048, n - 1), n - 1}
            mask = [int(i in eos) for i in range(n)]
            nodes = []
            scores = "x"
            if dtype == TensorProto.FLOAT16:
                nodes.append(helper.make_node("Cast", ["x"], ["f32"], to=TensorProto.FLOAT))
                scores = "f32"
            nodes.extend([helper.make_node("Where", ["eos", "lowest", scores], ["masked"]),
                          helper.make_node("ArgMax", ["masked"], ["y"], axis=1, keepdims=0, select_last_index=0)])
            graph = helper.make_graph(nodes, "cpu-exact-min-length",
                [helper.make_tensor_value_info("x", dtype, [rows, n])],
                [helper.make_tensor_value_info("y", TensorProto.INT64, [rows])],
                [helper.make_tensor("eos", TensorProto.BOOL, [n], mask),
                 helper.make_tensor("lowest", TensorProto.FLOAT, [1], [float(np.finfo(np.float32).min)])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
            onnx.checker.check_model(model)
            onnx.save(model, path)
            fixtures.append((label, rows, n, path))
            evidence.append(path)
    prefix = ROOT / "gpu-greedy-argmax-v4-mask-native-gpu4"
    commands = []
    for label, rows, n, path in fixtures:
        commands.append(shlex.join([workspace(binary), "compare-masked", workspace(provider), workspace(path),
            f"/evidence/webgpu-perf-oct20/{prefix.name}-{label}-rows{rows}-n{n}"]))
    command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
               "export ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE=1; taskset -c 48 " + "; taskset -c 48 ".join(commands))
    code, log = guarded(command, prefix)
    evidence.extend((prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")))
    if code != 0 or "FAIL" in log:
        raise RuntimeError(f"GPU min-length native CPU-ID comparison failed: {prefix}")
    counts = re.findall(r"^ARGMAX rows=(\d+) N=(\d+) comparisons=(\d+) mismatches=0$", log, re.M)
    profiles = re.findall(r"^PROFILE (.+)$", log, re.M)
    if len(counts) != 8 or len(profiles) != 8:
        raise RuntimeError("Incomplete GPU mask fixture coverage")
    results = []
    for fixture, count, profile in zip(fixtures, counts, profiles, strict=True):
        label, rows, n, _ = fixture
        if tuple(map(int, count)) != (rows, n, rows * 64):
            raise RuntimeError("Wrong GPU mask comparison count")
        path = host_path(profile)
        events = [e for e in json.loads(path.read_text()) if e.get("cat") == "Kernel"]
        if sum("&Where&" in e.get("name", "") for e in events) != 64:
            raise RuntimeError("Missing GPU Where dispatches")
        if profile_counts(path) != {"ArgMaxParallelPartial": 64, "ArgMaxParallelFinish": 64}:
            raise RuntimeError("Missing selected GPU ArgMax dispatches")
        if sum("&Cast&" in e.get("name", "") for e in events) != (64 if label == "f16" else 0):
            raise RuntimeError("Wrong GPU Cast dispatches")
        evidence.append(path)
        results.append(dict(dtype=label, rows=rows, n=n, comparisons=rows * 64, mismatches=0,
                            where_dispatches=64, selected_argmax_dispatches_per_stage=64))
    result = dict(status="gpu_min_length_cast_where_argmax_cpu_ids_exact", shapes=results,
        gpu_uuid=UUID, includes_nonfinite_zero_tie_and_unmasked_subnormal_cases=True,
        mask_value="float32_lowest_not_negative_infinity", model_eligible=False,
        provider_sha256=manifest["provider_sha256"], runtime_sha256=manifest["runtime_sha256"],
        source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], shapes=results)), flush=True)


if __name__ == "__main__":
    main()
