import json
from pathlib import Path
import re

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-matrix-probe/build-v1")
FIXTURES = BUILD.parent / "fixtures-v1"
PREFIX = ROOT / "fp16-dot4-matrix-probe-v1-native-gpu7-20261101"


if __name__ == "__main__":
    output = ROOT / "fp16-dot4-matrix-probe-v1-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["source_sha256"].items():
        if sha(BUILD / name) != digest:
            raise RuntimeError(f"Changed build input: {name}")
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    comparator = BUILD / "dot4_compare"
    if sha(provider) != manifest["provider_sha256"]:
        raise RuntimeError("Provider changed")
    if sha(comparator) != manifest["diagnostic_runner_sha256"]:
        raise RuntimeError("Comparator changed")
    guard_path = PREFIX.with_suffix(".guard.json")
    log_path = PREFIX.with_suffix(".log")
    guard = json.loads(guard_path.read_text())
    log = log_path.read_text()
    counts = re.findall(r"^DOT4 comparisons=(\d+) mismatches=(\d+)$", log, re.M)
    if counts != [("8192", "3000")]:
        raise RuntimeError("Unexpected/missing original matrix diagnostic count")
    if guard.get("exit_code") != 2 or guard.get("status") != "failed":
        raise RuntimeError("Original comparator rejection not proved")
    if guard.get("gpu_uuid") != "GPU-0e70e969-bbfc-80c5-d582-88be134632ea":
        raise RuntimeError("Original selected GPU changed")
    if "FAIL" in log or "MISMATCH" not in log:
        raise RuntimeError("Runtime failure or missing numerical evidence")
    fixtures = sorted(FIXTURES.glob("seed*.onnx"))
    if [path.name for path in fixtures] != [f"seed{seed}.onnx" for seed in range(8)]:
        raise RuntimeError("Incomplete fixture suite")
    if any("/workspace/" + str(path.relative_to("/home/sayanshaw/repos"))
           not in guard["command"][-1] for path in fixtures):
        raise RuntimeError("Original guard omitted a fixture")
    evidence = [manifest_path, provider, comparator, guard_path, log_path, *fixtures,
                ROOT / "build_fp16_dot4_matrix_probe.py",
                ROOT / "fp16_dot4_matrix_compare.cpp",
                ROOT / "prepare_fp16_dot4_matrix_fixtures.py",
                ROOT / "test_fp16_dot4_matrix_probe.py", Path(__file__)]
    report = dict(status="held_bitwise_matrix_substitution_rejected",
        native_f16_result_matrix_execution_demonstrated=True,
        finite_bitwise_comparisons=8192, bitwise_mismatches=3000,
        signed_zero_comparisons_included=True, nonzero_mismatches_observed=True,
        model_eligible=False, default_enabled=False, model_timing_performed=False,
        quality_qualified=False, provider_sha256=manifest["provider_sha256"],
        source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
        ("status", "finite_bitwise_comparisons", "bitwise_mismatches")}), flush=True)
