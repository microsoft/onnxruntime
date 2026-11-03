import json
from pathlib import Path
import re
import shlex
import subprocess

from build_fp16_dot4_pair_accumulator import FLAG
from build_qmoe_node_replay import sha
from prepare_fp16_dot4_dynamic_sweep import OUTPUT, prepare
from run_fp16_dot4_cross_shader import checked_provider, ORIGINAL
from run_fp16_dot4_tree_probe import GUARD, workspace, UUID


ROOT = Path(__file__).parent
DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-pair-accumulator")
BUILD = DIRECTORY / "dynamic-compare-v1"


def run_guard(command, prefix):
    for suffix in (".log", ".guard.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    result = subprocess.run(["python3", str(GUARD), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command])
    proof = json.loads(prefix.with_suffix(".guard.json").read_text())
    if proof["status"] != "completed":
        raise RuntimeError(f"GPU guard status={proof['status']}; preserved {prefix.with_suffix('.guard.json')}")
    if proof["gpu_uuid"] != UUID or proof["exit_code"] != result.returncode:
        raise RuntimeError("GPU guard device/child exit mismatch")
    return result.returncode, prefix.with_suffix(".log").read_text()


if __name__ == "__main__":
    outcome = ROOT / "fp16-dot4-dynamic-sweep-v1-outcome.json"
    execution = ROOT / "fp16-dot4-dynamic-sweep-v1-execution.json"
    if outcome.exists() or execution.exists():
        raise FileExistsError("Existing dynamic sweep evidence")
    primitive_path = ROOT / "fp16-dot4-pair-accumulator-v1-outcome.json"
    primitive = json.loads(primitive_path.read_text())
    if primitive["status"] != "seeded_pair_accumulator_diagnostic_supported":
        raise RuntimeError("Primitive prerequisite not supported")
    providers = [checked_provider(ORIGINAL), checked_provider(DIRECTORY)]
    binary = BUILD / "dynamic_sweep_compare"
    build_path = BUILD / "build-manifest.json"
    build = json.loads(build_path.read_text())
    if sha(binary) != build["comparator_sha256"] or sha(BUILD / "dynamic_sweep_compare.cpp") != build["source_sha256"]:
        raise RuntimeError("Changed dynamic comparator")
    dataset = prepare()
    dataset_path = OUTPUT / "manifest.json"
    original_capture = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-cross-shader/reference-dot4.f16")
    if sha(original_capture) != primitive["source_and_evidence_sha256"][str(original_capture)]:
        raise RuntimeError("Changed sealed reference")
    state = dict(status="running", records=[])
    execution.write_text(json.dumps(state, indent=2) + "\n")
    evidence = [Path(__file__), ROOT / "prepare_fp16_dot4_dynamic_sweep.py",
                ROOT / "test_fp16_dot4_dynamic_sweep.py",
                ROOT / "build_fp16_dot4_dynamic_sweep.py",
                ROOT / "fp16_dot4_dynamic_sweep_compare.cpp", GUARD,
                primitive_path, dataset_path, build_path, binary, original_capture, *providers]
    head = "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID
    stages = {}
    held = None
    for stage, elements in (("micro", 2048), ("sweep", 256000)):
        capture = OUTPUT / f"{stage}-reference.f16"
        if capture.exists():
            raise FileExistsError(capture)
        logs = []
        capture_digest = None
        for index, (mode, flag) in enumerate((
                ("capture", "ORT_WEBGPU_FP16_DOT4_TREE_ENUM"), ("compare", FLAG))):
            model = OUTPUT / ("reference.onnx" if index == 0 else "candidate.onnx")
            inputs = OUTPUT / f"{stage}.f16s"
            for path in (model, inputs):
                if sha(path) != dataset["output_sha256"][str(path)]:
                    raise RuntimeError(f"Changed dynamic input: {path}")
            subprocess.run(["nvidia-smi",
                "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader"], check=True)
            prefix = ROOT / f"fp16-dot4-dynamic-sweep-v1-{stage}-{mode}-gpu7"
            command = head + f"; export {flag}=1; exec " + shlex.join(
                [workspace(binary), mode, workspace(providers[index]), workspace(model),
                 workspace(inputs), workspace(capture)])
            code, log = run_guard(command, prefix)
            state["records"].append(dict(stage=stage, mode=mode, exit_code=code, prefix=str(prefix),
                                          log_sha256=sha(prefix.with_suffix(".log"))))
            execution.write_text(json.dumps(state, indent=2) + "\n")
            evidence.extend((model, inputs, prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")))
            if mode == "capture" and code == 1 and "Original enumeration exact pattern no longer matches" in log:
                held = "held_dynamic_baseline_pair_tree"
                break
            if code not in ((0,) if mode == "capture" else (0, 2)) or "FAIL" in log:
                raise RuntimeError(f"Dynamic {stage}/{mode} failed: {prefix.with_suffix('.log')}")
            if mode == "capture":
                if f"CAPTURE comparisons={elements * 4} elements={elements}" not in log or capture.stat().st_size != elements * 2:
                    raise RuntimeError("Incomplete dynamic capture")
                capture_digest = sha(capture)
                evidence.append(capture)
                if stage == "micro" and capture_digest != sha(original_capture):
                    held = "held_dynamic_reference_differs_from_sealed_capture"
                    break
            elif sha(capture) != capture_digest:
                raise RuntimeError("Dynamic reference changed during comparison")
            logs.append(log)
        if held:
            break
        entries = re.findall(r"^CROSS kind=(\d+) comparisons=(\d+) mismatches=(\d+)$", logs[1], re.M)
        if len(entries) != 5 or {int(k) for k, _, _ in entries} != set(range(5)):
            raise RuntimeError("Incomplete dynamic comparison")
        counts = {int(k): dict(comparisons=int(n), mismatches=int(m)) for k, n, m in entries}
        if any(v["comparisons"] != elements * 4 or not 0 <= v["mismatches"] <= elements * 4 for v in counts.values()):
            raise RuntimeError("Invalid dynamic coverage")
        if code != (0 if counts[1]["mismatches"] == 0 else 2):
            raise RuntimeError("Dynamic comparator exit disagrees")
        path_counts = re.findall(r"^PATH both_native=(\d+) corrected_pairs=(\d+)$", logs[1], re.M)
        if len(path_counts) != 1:
            raise RuntimeError("Incomplete correction evidence")
        native, corrected = map(int, path_counts[0])
        if not 0 < native < elements * 4 or not 0 < corrected <= elements * 8:
            raise RuntimeError("Native/correction paths not both exercised")
        stages[stage] = dict(counts=counts, both_pairs_native=native, corrected_pairs=corrected)
        print(json.dumps(dict(stage=stage, **stages[stage])), flush=True)
        if counts[1]["mismatches"] != 0:
            held = f"held_dynamic_{stage}_matrix_mismatches"
            break
    complete = "sweep" in stages
    state["status"] = held or "completed"
    execution.write_text(json.dumps(state, indent=2) + "\n")
    evidence.append(execution)
    report = dict(status=held or "dynamic_finite_half_accumulator_sweep_supported",
                  stages=stages, small_reference_matches_sealed_capture="micro" in stages,
                  full_finite_half_sweep_completed=complete, finite_patterns_dataset=63488,
                  changing_inputs_tested=bool(stages), sessions_per_phase=1, repeats=4,
                  full_fma_operand_space_covered=False, actual_model_weights_used=False,
                  default_enabled=False, model_eligible=False, model_timing_performed=False,
                  source_and_evidence_sha256={str(p): sha(p) for p in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "stages", "full_finite_half_sweep_completed")}), flush=True)
