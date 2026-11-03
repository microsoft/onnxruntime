import json
from pathlib import Path
import re
import shlex
import subprocess

from build_fp16_dot4_pair_accumulator import FLAG
from build_qmoe_node_replay import sha
from run_fp16_dot4_cross_shader import checked_provider
from run_fp16_dot4_tree_probe import guarded, workspace, UUID


ROOT = Path(__file__).parent
DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-pair-accumulator")
CROSS = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-cross-shader")
FIXTURES = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-pair-components/fixtures-v1")


if __name__ == "__main__":
    outcome = ROOT / "fp16-dot4-pair-accumulator-v1-outcome.json"
    if outcome.exists():
        raise FileExistsError(outcome)
    cross_path = ROOT / "fp16-dot4-cross-shader-v1-outcome.json"
    cross = json.loads(cross_path.read_text())
    if cross["status"] != "cross_shader_guarded_matrix_supported":
        raise RuntimeError("Cross-shader prerequisite not supported")
    binary = CROSS / "build-v1/cross_shader_compare"
    capture = CROSS / "reference-dot4.f16"
    if sha(binary) != cross["source_and_evidence_sha256"][str(binary)] or sha(capture) != cross["source_and_evidence_sha256"][str(capture)]:
        raise RuntimeError("Changed cross-shader comparator/reference")
    provider = checked_provider(DIRECTORY)
    fixture_proof_path = ROOT / "fp16-dot4-pair-components-v1-outcome.json"
    fixture_proof = json.loads(fixture_proof_path.read_text())["source_and_evidence_sha256"]
    paths = [FIXTURES / f"seed{seed}.onnx" for seed in range(8)]
    for path in paths:
        if sha(path) != fixture_proof[str(path)]:
            raise RuntimeError(f"Changed fixture: {path}")
    subprocess.run(["nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader"], check=True)
    head = "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID
    tail = "; exec " + shlex.join([workspace(binary), "compare", workspace(provider), workspace(capture), *map(workspace, paths)])
    prefix = ROOT / "fp16-dot4-pair-accumulator-v1-native-gpu7"
    code, log = guarded(head + f"; export {FLAG}=1" + tail, prefix)
    if code not in (0, 2) or "FAIL" in log:
        raise RuntimeError(f"Accumulator diagnostic failed: {prefix.with_suffix('.log')}")
    entries = re.findall(r"^CROSS kind=(\d+) comparisons=(\d+) mismatches=(\d+)$", log, re.M)
    if len(entries) != 5 or {int(k) for k, _, _ in entries} != set(range(5)):
        raise RuntimeError("Incomplete accumulator counts")
    counts = {int(k): dict(comparisons=int(n), mismatches=int(m)) for k, n, m in entries}
    if any(v["comparisons"] != 8192 or not 0 <= v["mismatches"] <= 8192 for v in counts.values()):
        raise RuntimeError("Invalid accumulator coverage")
    path_counts = re.findall(r"^PATH both_native=(\d+) corrected_pairs=(\d+)$", log, re.M)
    if len(path_counts) != 1:
        raise RuntimeError("Incomplete native/correction evidence")
    native, corrected = map(int, path_counts[0])
    if not 0 < native < 8192 or not 0 < corrected <= 16384:
        raise RuntimeError("Native and correction paths not both exercised")
    supported = counts[1]["mismatches"] == 0
    if code != (0 if supported else 2):
        raise RuntimeError("Comparator exit disagrees")
    invalid = ROOT / "fp16-dot4-pair-accumulator-v1-invalid-gpu7"
    invalid_code, invalid_log = guarded(head + f"; export {FLAG}=2" + tail, invalid)
    if invalid_code != 1 or f"{FLAG} must be 0 or 1" not in invalid_log:
        raise RuntimeError("Fresh invalid selector not rejected")
    if sha(capture) != cross["source_and_evidence_sha256"][str(capture)]:
        raise RuntimeError("Reference capture changed")
    evidence = [Path(__file__), ROOT / "build_fp16_dot4_pair_accumulator.py",
                ROOT / "test_fp16_dot4_pair_accumulator.py", cross_path, fixture_proof_path,
                DIRECTORY / "build-v1/build-manifest.json", provider, capture, binary, *paths]
    evidence.extend(p.with_suffix(s) for p in (prefix, invalid) for s in (".log", ".guard.json"))
    report = dict(status="seeded_pair_accumulator_diagnostic_supported" if supported else "held_seeded_pair_accumulator_not_bitwise",
                  counts=counts, both_pairs_native=native, corrected_pairs=corrected,
                  native_matrix_rows=16, accumulator_input="materialized_fp16_products_as_fp32",
                  independent_reference=True, invalid_selector_explicitly_rejected=True,
                  default_enabled=False, model_eligible=False, model_timing_performed=False,
                  source_and_evidence_sha256={str(p): sha(p) for p in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "counts", "both_pairs_native", "corrected_pairs")}), flush=True)
