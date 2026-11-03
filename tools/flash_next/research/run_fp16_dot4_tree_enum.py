import json
from pathlib import Path
import re
import shlex
import subprocess

from build_fp16_dot4_tree_enum import FLAG, PATTERNS
from build_qmoe_node_replay import sha
from prepare_fp16_dot4_tree_enum_fixtures import OUTPUT, prepare
from run_fp16_dot4_tree_probe import guarded, workspace, UUID


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-tree-enum/build-v1")
PREFIX = ROOT / "fp16-dot4-tree-enum-v1-native-gpu7"


if __name__ == "__main__":
    outcome_path = ROOT / "fp16-dot4-tree-enum-v1-outcome.json"
    if outcome_path.exists():
        raise FileExistsError(outcome_path)
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["source_sha256"].items():
        if sha(BUILD / name) != digest:
            raise RuntimeError(f"Changed build input: {name}")
    provider, runner = BUILD / "libonnxruntime_providers_webgpu.so", BUILD / "dot4_tree_compare"
    if sha(provider) != manifest["provider_sha256"] or sha(runner) != manifest["diagnostic_runner_sha256"]:
        raise RuntimeError("Changed provider or diagnostic runner")
    prepare()
    paths = [OUTPUT / f"seed{seed}.onnx" for seed in range(8)]
    subprocess.run(["nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader"], check=True)
    head = "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID
    tail = "; exec " + shlex.join([workspace(runner), workspace(provider), *map(workspace, paths)])
    code, log = guarded(head + f"; export {FLAG}=1" + tail, PREFIX)
    if code not in (0, 2) or "FAIL" in log:
        raise RuntimeError(f"Diagnostic failed with {code}; preserved {PREFIX.with_suffix('.log')}")
    counts = re.findall(r"^TREE pattern=(\d+) comparisons=(\d+) mismatches=(\d+)$", log, re.M)
    if len(counts) != PATTERNS or {int(p) for p, _, _ in counts} != set(range(PATTERNS)):
        raise RuntimeError("Incomplete or duplicate enumeration results")
    patterns = {int(p): dict(comparisons=int(total), mismatches=int(mismatch))
                for p, total, mismatch in counts}
    if any(v["comparisons"] != 8192 or not 0 <= v["mismatches"] <= 8192 for v in patterns.values()):
        raise RuntimeError("Wrong diagnostic coverage")
    supported = [p for p, value in patterns.items() if value["mismatches"] == 0]
    if code != (0 if supported else 2):
        raise RuntimeError("Comparator exit disagrees with enumeration")
    invalid_prefix = ROOT / "fp16-dot4-tree-enum-v1-invalid-gpu7"
    invalid_code, invalid_log = guarded(head + f"; export {FLAG}=2" + tail, invalid_prefix)
    if invalid_code != 1 or f"{FLAG} must be 0 or 1" not in invalid_log:
        raise RuntimeError("Fresh invalid selector was not explicitly rejected")
    evidence = [manifest_path, provider, runner, OUTPUT / "manifest.json", *paths,
                PREFIX.with_suffix(".guard.json"), PREFIX.with_suffix(".log"),
                invalid_prefix.with_suffix(".guard.json"), invalid_prefix.with_suffix(".log"),
                ROOT / "build_fp16_dot4_tree_enum.py", Path(__file__),
                ROOT / "prepare_fp16_dot4_tree_enum_fixtures.py",
                ROOT / "test_fp16_dot4_tree_enum.py", ROOT / "build_fp16_dot4_tree_probe.py",
                ROOT / "fp16_dot4_tree_compare.cpp", ROOT / "build_scoped_webgpu_source.py",
                ROOT / "run_fp16_dot4_tree_probe.py", ROOT / "shared_activation_guard.py"]
    report = dict(status="paired_dot4_diagnostic_patterns_supported" if supported else "held_no_exact_paired_dot4_tree",
        patterns=patterns, supported_patterns=supported, pairings=3, pair_rounding_choices=4,
        materialization_modes=["expression", "workgroup_fp16_pairs"],
        prior_four_pattern_indices=[36, 39, 40, 43],
        unique_fixture_elements_per_pattern=2048, repeats=4, comparisons_per_pattern=8192,
        total_finite_bitwise_comparisons=PATTERNS * 8192,
        invalid_selector_explicitly_rejected=True, default_enabled=False, model_eligible=False,
        model_timing_performed=False, quality_qualified=False, headline_eligible=False,
        provider_sha256=manifest["provider_sha256"],
        source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], supported_patterns=supported,
        best_patterns=sorted(patterns.items(), key=lambda item: item[1]["mismatches"])[:8],
        prior_four_counts={p: patterns[p] for p in report["prior_four_pattern_indices"]})), flush=True)
