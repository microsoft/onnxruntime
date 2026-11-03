import json
from pathlib import Path
import re
import shlex
import subprocess

from build_fp16_dot4_matrix_f32_probe import FLAG
from build_qmoe_node_replay import sha
from run_fp16_dot4_tree_probe import guarded, workspace, UUID


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-matrix-f32-probe/build-v1")
FIXTURES = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-matrix-probe/fixtures-v1")
PREFIX = ROOT / "fp16-dot4-matrix-f32-probe-v1-native-gpu7"


if __name__ == "__main__":
    outcome_path = ROOT / "fp16-dot4-matrix-f32-probe-v1-outcome.json"
    if outcome_path.exists():
        raise FileExistsError(outcome_path)
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest["source_sha256"].items():
        if sha(BUILD / name) != digest:
            raise RuntimeError(f"Changed build input: {name}")
    provider, runner = BUILD / "libonnxruntime_providers_webgpu.so", BUILD / "dot4_compare"
    if sha(provider) != manifest["provider_sha256"] or sha(runner) != manifest["diagnostic_runner_sha256"]:
        raise RuntimeError("Changed provider or diagnostic runner")
    fixture_manifest = FIXTURES / "manifest.json"
    records = json.loads(fixture_manifest.read_text())["fixtures"]
    paths = [FIXTURES / f"seed{seed}.onnx" for seed in range(8)]
    if len(records) != 8 or {Path(r["path"]).name for r in records} != {p.name for p in paths}:
        raise RuntimeError("Incomplete original diagnostic fixtures")
    for path in paths:
        record = next(r for r in records if Path(r["path"]).name == path.name)
        if sha(path) != record["sha256"]:
            raise RuntimeError(f"Changed original fixture: {path}")
    subprocess.run(["nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader"], check=True)
    head = "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID
    tail = "; exec " + shlex.join([workspace(runner), workspace(provider), *map(workspace, paths)])
    code, log = guarded(head + f"; export {FLAG}=1" + tail, PREFIX)
    if code not in (0, 2) or "FAIL" in log:
        raise RuntimeError(f"Diagnostic runtime failed with {code}; preserved {PREFIX.with_suffix('.log')}")
    counts = re.findall(r"^DOT4 comparisons=(\d+) mismatches=(\d+)$", log, re.M)
    classified = re.findall(r"^DOT4 classifications zero_sign_mismatches=(\d+) nonzero_mismatches=(\d+)$", log, re.M)
    if len(counts) != 1 or len(classified) != 1:
        raise RuntimeError("Missing or duplicate diagnostic counts")
    comparisons, mismatches = map(int, counts[0])
    zero_sign, nonzero = map(int, classified[0])
    if comparisons != 8192 or not 0 <= mismatches <= comparisons or mismatches != zero_sign + nonzero:
        raise RuntimeError("Wrong diagnostic coverage or classification")
    if code != (2 if mismatches else 0):
        raise RuntimeError("Comparator exit disagrees with bitwise result")
    invalid_prefix = ROOT / "fp16-dot4-matrix-f32-probe-v1-invalid-gpu7"
    invalid_code, invalid_log = guarded(head + f"; export {FLAG}=2" + tail, invalid_prefix)
    if invalid_code != 1 or f"{FLAG} must be 0 or 1" not in invalid_log:
        raise RuntimeError("Fresh invalid selector was not explicitly rejected")
    evidence = [manifest_path, provider, runner, fixture_manifest, *paths,
                PREFIX.with_suffix(".guard.json"), PREFIX.with_suffix(".log"),
                invalid_prefix.with_suffix(".guard.json"), invalid_prefix.with_suffix(".log"),
                ROOT / "build_fp16_dot4_matrix_f32_probe.py", Path(__file__),
                ROOT / "test_fp16_dot4_matrix_f32_probe.py", ROOT / "build_fp16_dot4_matrix_probe.py",
                ROOT / "fp16_dot4_matrix_compare.cpp", ROOT / "build_scoped_webgpu_source.py",
                ROOT / "run_fp16_dot4_tree_probe.py", ROOT / "shared_activation_guard.py"]
    outcome = dict(status=("held_single_f32_matrix_dot4_not_bitwise" if mismatches else
                           "single_f32_matrix_dot4_diagnostic_supported_not_model_qualified"),
        native_f16_input_f32_result_matrix_execution_demonstrated=True,
        finite_bitwise_comparisons=comparisons, unique_fixture_elements=2048, repeats=4,
        bitwise_mismatches=mismatches, zero_sign_mismatches=zero_sign, nonzero_mismatches=nonzero,
        single_native_matrix_accumulation=True, explicit_fp32_product_materialization=False,
        model_eligible=False, default_enabled=False, model_timing_performed=False,
        invalid_selector_explicitly_rejected=True, quality_qualified=False, headline_eligible=False,
        gpu_uuid=UUID, provider_sha256=manifest["provider_sha256"],
        source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome_path.write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps({key: outcome[key] for key in
        ("status", "finite_bitwise_comparisons", "bitwise_mismatches", "zero_sign_mismatches",
         "nonzero_mismatches", "invalid_selector_explicitly_rejected")}), flush=True)
