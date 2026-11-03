import json
from pathlib import Path
import re
import shlex
import subprocess

from build_fp16_dot4_pair_accumulator import FLAG
from build_qmoe_node_replay import sha
from prepare_fp16_dot4_accumulator_sweep import OUTPUT, prepare
from run_fp16_dot4_cross_shader import checked_provider, ORIGINAL
from run_fp16_dot4_tree_probe import guarded, workspace, UUID


ROOT = Path(__file__).parent
DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-fp16-dot4-pair-accumulator")
COMPARE = DIRECTORY / "stress-compare-v1"


if __name__ == "__main__":
    outcome = ROOT / "fp16-dot4-accumulator-sweep-v1-outcome.json"
    execution = ROOT / "fp16-dot4-accumulator-sweep-v1-execution.json"
    if outcome.exists() or execution.exists():
        raise FileExistsError("Existing stress execution")
    prerequisite_path = ROOT / "fp16-dot4-pair-accumulator-v1-outcome.json"
    prerequisite = json.loads(prerequisite_path.read_text())
    if prerequisite["status"] != "seeded_pair_accumulator_diagnostic_supported":
        raise RuntimeError("Seeded-accumulator prerequisite not supported")
    providers = [checked_provider(ORIGINAL), checked_provider(DIRECTORY)]
    binary = COMPARE / "sweep_compare"
    build_path = COMPARE / "build-manifest.json"
    build = json.loads(build_path.read_text())
    if sha(binary) != build["comparator_sha256"] or sha(COMPARE / "sweep_compare.cpp") != build["source_sha256"]:
        raise RuntimeError("Changed sweep comparator")
    dataset = prepare()
    dataset_path = OUTPUT / "manifest.json"
    state = dict(status="running", phases=[])
    execution.write_text(json.dumps(state, indent=2) + "\n")
    capture = OUTPUT / "reference-dot4.f16"
    if capture.exists():
        raise FileExistsError(capture)
    evidence = [Path(__file__), ROOT / "prepare_fp16_dot4_accumulator_sweep.py",
                ROOT / "test_fp16_dot4_accumulator_sweep.py",
                ROOT / "build_fp16_dot4_sweep_compare.py",
                prerequisite_path, dataset_path, build_path, binary, *providers]
    head = "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID
    capture_digest = None
    logs = []
    baseline_failed = False
    for index, (mode, flag) in enumerate((
            ("capture", "ORT_WEBGPU_FP16_DOT4_TREE_ENUM"), ("compare", FLAG))):
        paths = []
        for batch in dataset["batches"]:
            for record in batch["records"]:
                value = record["fixtures"][index]
                path = Path(value["path"])
                if sha(path) != value["sha256"]:
                    raise RuntimeError(f"Changed stress fixture: {path}")
                paths.append(path)
        if len(paths) != 1000:
            raise RuntimeError("Wrong sweep fixture count")
        listing = OUTPUT / f"{mode}-fixtures.txt"
        with listing.open("x") as stream:
            stream.write("".join(workspace(p) + "\n" for p in paths))
        evidence.extend([listing, *paths])
        subprocess.run(["nvidia-smi",
            "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader"], check=True)
        prefix = ROOT / f"fp16-dot4-accumulator-sweep-v1-{mode}-gpu7"
        command = head + f"; export {flag}=1; exec " + shlex.join(
            [workspace(binary), mode, workspace(providers[index]), workspace(capture), workspace(listing)])
        code, log = guarded(command, prefix)
        state["phases"].append(dict(mode=mode, exit_code=code, prefix=str(prefix),
                                    log_sha256=sha(prefix.with_suffix(".log"))))
        execution.write_text(json.dumps(state, indent=2) + "\n")
        evidence.extend((prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")))
        if mode == "capture" and code == 1 and "Original enumeration exact pattern no longer matches" in log:
            baseline_failed = True
            break
        if code not in ((0,) if mode == "capture" else (0, 2)) or "FAIL" in log:
            raise RuntimeError(f"Stress {mode} failed: {prefix.with_suffix('.log')}")
        if mode == "capture":
            if "CAPTURE comparisons=1024000 elements=256000" not in log or capture.stat().st_size != 512000:
                raise RuntimeError("Incomplete sweep reference")
            capture_digest = sha(capture)
            evidence.append(capture)
            print(json.dumps(dict(phase=mode, comparisons=1024000, reference_bytes=512000)), flush=True)
        elif sha(capture) != capture_digest:
            raise RuntimeError("Sweep reference changed during comparison")
        logs.append(log)
    counts = None
    native = corrected = None
    if not baseline_failed:
        entries = re.findall(r"^CROSS kind=(\d+) comparisons=(\d+) mismatches=(\d+)$", logs[1], re.M)
        if len(entries) != 5 or {int(k) for k, _, _ in entries} != set(range(5)):
            raise RuntimeError("Incomplete sweep comparisons")
        counts = {int(k): dict(comparisons=int(n), mismatches=int(m)) for k, n, m in entries}
        if any(v["comparisons"] != 1024000 or not 0 <= v["mismatches"] <= 1024000 for v in counts.values()):
            raise RuntimeError("Invalid sweep counts")
        if code != (0 if counts[1]["mismatches"] == 0 else 2):
            raise RuntimeError("Sweep comparator exit disagrees")
        paths = re.findall(r"^PATH both_native=(\d+) corrected_pairs=(\d+)$", logs[1], re.M)
        if len(paths) != 1:
            raise RuntimeError("Incomplete sweep correction evidence")
        native, corrected = map(int, paths[0])
        if not 0 < native < 1024000 or not 0 < corrected <= 2048000:
            raise RuntimeError("Sweep did not exercise native and corrected paths")
    state["status"] = "held_baseline_tree_failure" if baseline_failed else "completed"
    execution.write_text(json.dumps(state, indent=2) + "\n")
    evidence.append(execution)
    report = dict(status="held_baseline_pair_tree_not_transferable" if baseline_failed else
                  ("finite_half_accumulator_sweep_supported" if counts[1]["mismatches"] == 0
                   else "held_finite_half_accumulator_sweep_mismatches"),
                  counts=counts, full_finite_half_sweep_completed=not baseline_failed,
                  finite_half_patterns_dataset=63488, edge_case_batch_completed=not baseline_failed,
                  both_pairs_native=native, corrected_pairs=corrected, repeats=4,
                  full_fma_operand_space_covered=False, actual_model_weights_used=False,
                  default_enabled=False, model_eligible=False, model_timing_performed=False,
                  source_and_evidence_sha256={str(p): sha(p) for p in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "counts",
        "full_finite_half_sweep_completed", "both_pairs_native", "corrected_pairs")}), flush=True)
