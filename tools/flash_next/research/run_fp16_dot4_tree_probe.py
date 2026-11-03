import json
from pathlib import Path
import re
import shlex
import subprocess

from build_fp16_dot4_tree_probe import FLAG
from build_qmoe_node_replay import sha
from prepare_fp16_dot4_tree_fixtures import OUTPUT, prepare


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-tree-probe/build-v1")
PREFIX = ROOT / "fp16-dot4-tree-probe-v1-native-gpu7"
GUARD = ROOT / "shared_activation_guard.py"
UUID = "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"


def workspace(path):
    return "/workspace/" + str(path.relative_to("/home/sayanshaw/repos"))


def guarded(command, prefix):
    for suffix in (".guard.json", ".log"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    result = subprocess.run(["python3", str(GUARD), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "180",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command])
    proof = json.loads(prefix.with_suffix(".guard.json").read_text())
    if proof.get("gpu_uuid") != UUID or proof.get("exit_code") != result.returncode:
        raise RuntimeError("Guard did not prove selected GPU/child exit")
    return result.returncode, prefix.with_suffix(".log").read_text()


if __name__ == "__main__":
    outcome_path = ROOT / "fp16-dot4-tree-probe-v1-outcome.json"
    if outcome_path.exists():
        raise FileExistsError(outcome_path)
    subprocess.run(["nvidia-smi",
        "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader"], check=True)
    prepare()
    manifest = json.loads((BUILD / "build-manifest.json").read_text())
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    runner = BUILD / "dot4_tree_compare"
    if sha(provider) != manifest["provider_sha256"]:
        raise RuntimeError("Provider hash changed")
    if sha(runner) != manifest["diagnostic_runner_sha256"]:
        raise RuntimeError("Runner hash changed")
    paths = [OUTPUT / f"seed{seed}.onnx" for seed in range(8)]
    head = "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID
    tail = "; exec " + shlex.join([workspace(runner), workspace(provider), *map(workspace, paths)])
    code, log = guarded(head + f"; export {FLAG}=1" + tail, PREFIX)
    if code not in (0, 2):
        raise RuntimeError(f"Diagnostic runtime failed with {code}; preserved {PREFIX.with_suffix('.log')}")
    counts = re.findall(r"^TREE pattern=(\d+) comparisons=(\d+) mismatches=(\d+)$", log, re.M)
    if len(counts) != 4 or {int(p) for p, _, _ in counts} != set(range(4)):
        raise RuntimeError("Missing/duplicate diagnostic pattern counts")
    patterns = {p: dict(comparisons=int(total), mismatches=int(mismatch))
                for p, total, mismatch in counts}
    if any(value["comparisons"] != 8192 or not 0 <= value["mismatches"] <= 8192
           for value in patterns.values()):
        raise RuntimeError("Wrong diagnostic coverage")
    supported = [p for p, value in patterns.items() if value["mismatches"] == 0]
    if code != (0 if supported else 2):
        raise RuntimeError("Comparator exit disagrees with diagnostic results")
    invalid_prefix = ROOT / "fp16-dot4-tree-probe-v1-invalid-gpu7"
    invalid_code, invalid_log = guarded(head + f"; export {FLAG}=2" + tail, invalid_prefix)
    if invalid_code != 1 or f"{FLAG} must be 0 or 1" not in invalid_log:
        raise RuntimeError("Invalid selector not explicitly rejected")
    evidence = [*paths, OUTPUT / "manifest.json", BUILD / "build-manifest.json",
                provider, runner, PREFIX.with_suffix(".log"), PREFIX.with_suffix(".guard.json"),
                invalid_prefix.with_suffix(".log"), invalid_prefix.with_suffix(".guard.json"),
                ROOT / "build_fp16_dot4_tree_probe.py",
                ROOT / "fp16_dot4_tree_compare.cpp", Path(__file__),
                ROOT / "prepare_fp16_dot4_tree_fixtures.py",
                ROOT / "test_fp16_dot4_tree_probe.py", GUARD, ROOT / "nvml_monitor.py"]
    outcome = dict(status="diagnostic_patterns_supported" if supported else "held_no_exact_pattern",
        model_eligible=False, default_enabled=False, model_timing_performed=False,
        signed_zero_comparisons_included=True, patterns=patterns, supported_patterns=supported,
        invalid_selector_explicitly_rejected=True, gpu_uuid=UUID,
        provider_sha256=manifest["provider_sha256"],
        source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome_path.write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2), flush=True)
