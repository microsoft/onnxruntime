import json
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np
import onnx
from onnx import numpy_helper

from build_qmoe_node_replay import sha
from run_fp16_dot4_tree_probe import guarded, workspace, UUID


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-cross-shader/build-v1")
ORIGINAL = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-tree-enum")
CANDIDATE = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-pair-components")


def checked_provider(directory):
    manifest = json.loads((directory / "build-v1/build-manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        if sha(directory / "build-v1" / name) != digest:
            raise RuntimeError(f"Changed build input: {name}")
    provider = directory / "build-v1/libonnxruntime_providers_webgpu.so"
    if sha(provider) != manifest["provider_sha256"]:
        raise RuntimeError(f"Changed provider: {provider}")
    return provider


if __name__ == "__main__":
    outcome = ROOT / "fp16-dot4-cross-shader-v1-outcome.json"
    capture = BUILD.parent / "reference-dot4.f16"
    if outcome.exists() or capture.exists():
        raise FileExistsError("Existing cross-shader evidence")
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    binary = BUILD / "cross_shader_compare"
    if sha(binary) != manifest["comparator_sha256"] or sha(BUILD / "cross_shader_compare.cpp") != manifest["source_sha256"]:
        raise RuntimeError("Changed comparator")
    old_provider, new_provider = map(checked_provider, (ORIGINAL, CANDIDATE))
    old_paths = [ORIGINAL / "fixtures-v1" / f"seed{seed}.onnx" for seed in range(8)]
    new_paths = [CANDIDATE / "fixtures-v1" / f"seed{seed}.onnx" for seed in range(8)]
    qualification_paths = [ROOT / "fp16-dot4-tree-enum-v1-outcome.json",
                           ROOT / "fp16-dot4-pair-components-v1-outcome.json"]
    for qualification, paths in zip(qualification_paths, (old_paths, new_paths)):
        recorded = json.loads(qualification.read_text())["source_and_evidence_sha256"]
        for path in paths:
            if sha(path) != recorded[str(path)]:
                raise RuntimeError(f"Changed qualified fixture: {path}")
    for old, new in zip(old_paths, new_paths):
        tensors = [{v.name: numpy_helper.to_array(v) for v in onnx.load(p).graph.initializer} for p in (old, new)]
        if tensors[0]["a"].tobytes() != tensors[1]["a"].tobytes() or any(
                not np.array_equal(tensors[0][name][:16], tensors[1][name][:16]) for name in ("b", "scales")):
            raise RuntimeError("Cross-shader fixture operands differ")
    head = "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID
    prefixes = [ROOT / f"fp16-dot4-cross-shader-v1-{mode}-gpu7" for mode in ("capture", "compare")]
    logs = []
    capture_digest = None
    for mode, provider, paths, flag, prefix in zip(
            ("capture", "compare"), (old_provider, new_provider), (old_paths, new_paths),
            ("ORT_WEBGPU_FP16_DOT4_TREE_ENUM", "ORT_WEBGPU_FP16_DOT4_PAIR_COMPONENTS"), prefixes):
        subprocess.run(["nvidia-smi",
            "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader"], check=True)
        command = head + f"; export {flag}=1; exec " + shlex.join(
            [workspace(binary), mode, workspace(provider), workspace(capture), *map(workspace, paths)])
        code, log = guarded(command, prefix)
        if code not in ((0,) if mode == "capture" else (0, 2)) or "FAIL" in log:
            raise RuntimeError(f"Cross-shader {mode} failed: {prefix.with_suffix('.log')}")
        logs.append(log)
        if mode == "capture" and ("CAPTURE comparisons=8192 elements=2048" not in log or capture.stat().st_size != 4096):
            raise RuntimeError("Incomplete original-reference capture")
        if mode == "capture":
            capture_digest = sha(capture)
        elif sha(capture) != capture_digest:
            raise RuntimeError("Reference capture changed during comparison")
    entries = re.findall(r"^CROSS kind=(\d+) comparisons=(\d+) mismatches=(\d+)$", logs[1], re.M)
    if len(entries) != 5 or {int(k) for k, _, _ in entries} != set(range(5)):
        raise RuntimeError("Missing cross-shader counts")
    counts = {int(k): dict(comparisons=int(n), mismatches=int(m)) for k, n, m in entries}
    if any(v["comparisons"] != 8192 or not 0 <= v["mismatches"] <= 8192 for v in counts.values()):
        raise RuntimeError("Invalid cross-shader coverage")
    path, = re.findall(r"^PATH both_native=(\d+) corrected_pairs=(\d+)$", logs[1], re.M)
    native, corrected = map(int, path)
    if not 0 < native < 8192 or not 0 < corrected <= 16384:
        raise RuntimeError("Native and corrected paths not both exercised")
    if code != (0 if counts[1]["mismatches"] == 0 else 2):
        raise RuntimeError("Comparator exit disagrees with guarded cross-shader result")
    evidence = [Path(__file__), ROOT / "build_fp16_dot4_cross_shader.py",
                ROOT / "fp16_dot4_cross_shader_compare.cpp", manifest_path, binary, capture,
                old_provider, new_provider, *old_paths, *new_paths, *qualification_paths]
    evidence.extend(p.with_suffix(s) for p in prefixes for s in (".log", ".guard.json"))
    report = dict(status="cross_shader_guarded_matrix_supported" if counts[1]["mismatches"] == 0 else "held_cross_shader_matrix_not_bitwise",
                  counts=counts, kind_names=["raw_matrix", "guarded_matrix", "scalar_f16_tree", "scalar_f32_tree", "candidate_local_dot"],
                  both_pairs_native=native, corrected_pairs=corrected,
                  original_enumeration_47_95_reverified=True, unique_elements=2048, repeats=4,
                  provider_changed=False, default_enabled=False, model_eligible=False, model_timing_performed=False,
                  source_and_evidence_sha256={str(p): sha(p) for p in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "counts", "kind_names", "both_pairs_native", "corrected_pairs")}), flush=True)
