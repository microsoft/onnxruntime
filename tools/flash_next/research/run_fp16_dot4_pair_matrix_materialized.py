import json
from pathlib import Path
import re
import shlex
import subprocess

from build_fp16_dot4_pair_matrix_materialized import FLAG
from build_qmoe_node_replay import sha
from prepare_fp16_dot4_pair_matrix_fixtures import prepare
from run_fp16_dot4_tree_probe import guarded, workspace, UUID


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-pair-matrix-materialized/build-v1")
FIXTURES = BUILD.parent / "fixtures-v1"
PREFIX = ROOT / "fp16-dot4-pair-matrix-materialized-v1-native-gpu7"


if __name__ == "__main__":
    outcome_path = ROOT / "fp16-dot4-pair-matrix-materialized-v1-outcome.json"
    if outcome_path.exists():
        raise FileExistsError(outcome_path)
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["source_sha256"].items():
        if sha(BUILD / name) != digest:
            raise RuntimeError(f"Changed build input: {name}")
    provider, runner = BUILD / "libonnxruntime_providers_webgpu.so", BUILD / "pair_matrix_compare"
    if sha(provider) != manifest["provider_sha256"] or sha(runner) != manifest["diagnostic_runner_sha256"]:
        raise RuntimeError("Changed provider or diagnostic runner")
    prepare(output=FIXTURES)
    paths = [FIXTURES / f"seed{seed}.onnx" for seed in range(8)]
    parent = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-pair-matrix-probe/fixtures-v1")
    for path in paths:
        if sha(path) != sha(parent / path.name):
            raise RuntimeError(f"Changed fixture bytes: {path}")
    subprocess.run(["nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader"], check=True)
    head = "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID
    tail = "; exec " + shlex.join([workspace(runner), workspace(provider), *map(workspace, paths)])
    code, log = guarded(head + f"; export {FLAG}=1" + tail, PREFIX)
    if code not in (0, 2) or "FAIL" in log:
        raise RuntimeError(f"Diagnostic failed with {code}; preserved {PREFIX.with_suffix('.log')}")
    counts = re.findall(r"^PAIR pattern=(\d+) comparisons=(\d+) mismatches=(\d+)$", log, re.M)
    if len(counts) != 2 or {int(p) for p, _, _ in counts} != {0, 1}:
        raise RuntimeError("Missing or duplicate pair results")
    patterns = {int(p): dict(comparisons=int(total), mismatches=int(mismatch))
                for p, total, mismatch in counts}
    if any(v["comparisons"] != 8192 or not 0 <= v["mismatches"] <= 8192 for v in patterns.values()):
        raise RuntimeError("Wrong diagnostic coverage")
    matches = re.findall(r"^PATH both_native=(\d+) corrected_pairs=(\d+) small_pairs=(\d+) boundary_pairs=(\d+)$", log, re.M)
    if len(matches) != 1:
        raise RuntimeError("Incomplete native/correction path evidence")
    both_native, corrected_pairs, small_pairs, boundary_pairs = map(int, matches[0])
    if not 0 < both_native < 8192 or not 0 < corrected_pairs <= 16384:
        raise RuntimeError("Native and correction paths were not both exercised")
    if not 0 <= small_pairs <= corrected_pairs or not 0 <= boundary_pairs <= corrected_pairs:
        raise RuntimeError("Invalid correction counts")
    supported = patterns[1]["mismatches"] == 0
    if code != (0 if supported else 2):
        raise RuntimeError("Comparator exit disagrees with guarded result")
    invalid_prefix = ROOT / "fp16-dot4-pair-matrix-materialized-v1-invalid-gpu7"
    invalid_code, invalid_log = guarded(head + f"; export {FLAG}=2" + tail, invalid_prefix)
    if invalid_code != 1 or f"{FLAG} must be 0 or 1" not in invalid_log:
        raise RuntimeError("Fresh invalid selector was not explicitly rejected")
    evidence = [manifest_path, provider, runner, FIXTURES / "manifest.json", *paths,
                PREFIX.with_suffix(".guard.json"), PREFIX.with_suffix(".log"),
                invalid_prefix.with_suffix(".guard.json"), invalid_prefix.with_suffix(".log"),
                ROOT / "build_fp16_dot4_pair_matrix_materialized.py", Path(__file__),
                ROOT / "test_fp16_dot4_pair_matrix_materialized.py",
                ROOT / "build_fp16_dot4_pair_matrix_probe.py",
                ROOT / "fp16_dot4_pair_matrix_compare.cpp",
                ROOT / "prepare_fp16_dot4_pair_matrix_fixtures.py",
                ROOT / "build_fp16_dot4_matrix_f32_probe.py", ROOT / "build_scoped_webgpu_source.py",
                ROOT / "run_fp16_dot4_tree_probe.py", ROOT / "shared_activation_guard.py"]
    report = dict(status="materialized_guarded_pair_matrix_diagnostic_supported" if supported else "held_materialized_pair_matrix_not_bitwise",
        patterns=patterns, both_pairs_native_comparisons=both_native, corrected_pair_results=corrected_pairs,
        zero_subnormal_pair_results=small_pairs, boundary_pair_results=boundary_pairs,
        unique_fixture_elements_per_pattern=2048, repeats=4, comparisons_per_pattern=8192,
        fixture_bytes_match_v1=True, fp16_pair_materialization=True,
        invalid_selector_explicitly_rejected=True, default_enabled=False, model_eligible=False,
        model_timing_performed=False, quality_qualified=False, headline_eligible=False,
        provider_sha256=manifest["provider_sha256"],
        source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
        ("status", "patterns", "both_pairs_native_comparisons", "corrected_pair_results",
         "zero_subnormal_pair_results", "boundary_pair_results")}), flush=True)
