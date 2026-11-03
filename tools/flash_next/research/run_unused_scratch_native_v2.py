import json
import math
from pathlib import Path
import re
import statistics

from build_qmoe_node_replay import replace_once, sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_fc2_rank_waves_native import guarded
from run_gpu_slice_model import verify_sources


REPOS = Path("/home/sayanshaw/repos")
BUILD = REPOS / "qwen38-int4-qmoe-unused-scratch/build-v2"
NATIVE = BUILD.parent / "native-v2"
PARENT = REPOS / "qwen38-int4-fc2-private-a/build-v1"
FLAG = "ORT_WEBGPU_QMOE_LAZY_UNUSED_SCRATCH"
VARIANT = "qmoe-unused-scratch-v2"
TEMPLATE = "fc2-private-a-v1-native-flag0-gpu0"


def mounted(path):
    return "/workspace/" + str(Path(path).relative_to(REPOS))


def command_for(prefix, flag, replay, candidate, operator=False):
    if flag not in (0, 1, 2) or replay not in (0, 1):
        raise ValueError("Invalid scratch selector or replay mode")
    name = "fc2-private-a-v1-before-gpu0" if operator else TEMPLATE
    command = json.loads((ROOT / (name + ".guard.json")).read_text())["command"].copy()
    if command[:5] != ["taskset", "-c", "0-7", "bash", "-lc"]:
        raise RuntimeError("Unexpected owned native affinity")
    shell = command[-1]
    dump = f"/evidence/webgpu-perf-oct20/{TEMPLATE}-shaders.txt"
    shell = replace_once(shell, f"ORT_WEBGPU_EP_SHADER_DUMP_FILE={dump}", "ORT_WEBGPU_EP_SHADER_DUMP_FILE=")
    shell = replace_once(shell, "ORT_WEBGPU_QMOE_COMMAND_REPLAY=1", f"ORT_WEBGPU_QMOE_COMMAND_REPLAY={replay}")
    if candidate:
        shell = replace_once(shell, mounted(PARENT / "libonnxruntime_providers_webgpu.so"),
                             mounted(BUILD / "libonnxruntime_providers_webgpu.so"))
    if operator:
        shell = replace_once(shell, name + ".json", prefix.name + ".json")
    else:
        shell = replace_once(shell,
            "/workspace/qwen38-int4-qmoe-fc2-final-mix/build-v2/native_compare",
            mounted(NATIVE / "native_compare"))
    command[-1] = f"export {FLAG}={flag}; " + shell
    return command


def main():
    output = ROOT / f"{VARIANT}-native-operator-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    native_parent = ROOT / "fc2-private-a-v1-native-outcome.json"
    operator_parent = ROOT / "fc2-private-a-v1-operator-screen.json"
    verify_sources(native_parent, "fc2_private_a_native_finite_bitwise_and_shader_routing_qualified")
    verify_sources(operator_parent, "fc2_private_a_operator_screen_completed")
    build = json.loads((BUILD / "build-manifest.json").read_text())
    generation = json.loads((NATIVE / "generation.json").read_text())
    for records in (build["source_sha256"], generation["source_sha256"]):
        for path, digest in records.items():
            path = host_path(path) if path.startswith(("/workspace/", "/evidence/")) else Path(path)
            if sha(path) != digest:
                raise RuntimeError(f"Changed scratch source: {path}")
    if (not build["baseline_byte_identical"] or not build["original_qmoe_object_byte_identical"]
            or sha(BUILD / "libonnxruntime_providers_webgpu.so") != build["provider_sha256"]):
        raise RuntimeError("Missing original object/provider provenance")
    evidence = [Path(__file__), GUARD, native_parent, operator_parent, BUILD / "build-manifest.json",
                BUILD / "libonnxruntime_providers_webgpu.so", NATIVE / "generation.json", NATIVE / "native_compare",
                PARENT / "native_benchmark"]
    fingerprints = None
    phases = []
    for flag, replay, candidate, phase in (
            (0, 0, False, "original"), (0, 0, True, "default-off"),
            (1, 0, True, "selected"), (1, 1, True, "replay-fallback"),
            (2, 0, True, "invalid")):
        prefix = ROOT / f"{VARIANT}-native-{phase}-gpu0"
        command = command_for(prefix, flag, replay, candidate)
        child, guard = guarded(prefix, command)
        text = prefix.with_suffix(".log").read_text()
        evidence.extend(prefix.with_suffix(s) for s in (".log", ".guard.json"))
        if flag == 2:
            if child.returncode != 1 or f"{FLAG} must be 0 or 1" not in text:
                raise RuntimeError("Invalid process-static scratch selector did not reject explicitly")
            phases.append({"phase": phase, "invalid_selector_rejected": True})
            continue
        child.check_returncode()
        rows = re.findall(r"^PASS changing_inputs_and_routes (\S+) rounds=32 invalid_flag_rejected=1 output_fingerprint=(\d+)$",
                          text, flags=re.MULTILINE)
        if (guard["status"] != "completed" or len(rows) != 23 or "FAIL" in text
                or "PASS total_exact_elements=9748480; CPU fallback disabled" not in text):
            raise RuntimeError("Incomplete fused/unfused arithmetic and fallback qualification")
        if fingerprints is None:
            fingerprints = rows
        if rows != fingerprints:
            raise RuntimeError("Independent original/default/selected/replay output bytes differ")
        phases.append({"phase": phase, "flag": flag, "command_replay": replay,
                       "finite_bitwise_elements": 9748480, "fixtures": 23,
                       "all_output_fingerprints_agree": True,
                       "sampled_peak_gpu_gib": guard["peak_gpu_used_bytes"] / 2**30})
    native = {
        "status": "qmoe_unused_scratch_native_cross_provider_qualified",
        "provider_sha256": build["provider_sha256"],
        "baseline_provider_sha256": build["baseline_provider_sha256"],
        "phases": phases,
        "quality_qualified": False,
        "performance_qualified": False,
        "source_and_evidence_sha256": {str(path): sha(path) for path in evidence},
    }
    native_path = ROOT / f"{VARIANT}-native.qualification.json"
    with native_path.open("x") as stream:
        json.dump(native, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in native.items()
                      if key != "source_and_evidence_sha256"}), flush=True)
    times = []
    output_hash = None
    peaks = []
    for flag, phase in ((0, "before"), (1, "candidate"), (0, "after")):
        prefix = ROOT / f"{VARIANT}-operator-{phase}-gpu0"
        command = command_for(prefix, flag, 0, True, operator=True)
        child, guard = guarded(prefix, command)
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        if (guard["status"] != "completed" or raw["status"] != "qmoe_operator_benchmark_completed"
                or raw["warmups"] != 16 or raw["measured_runs"] != 128
                or raw["profiling"] or raw["measured_major_faults"] != 0
                or len(raw["run_seconds"]) != 128
                or any(not math.isfinite(value) or value <= 0 for value in raw["run_seconds"])):
            raise RuntimeError("Incomplete, invalid or faulting native operator bracket")
        if output_hash is None:
            output_hash = raw["output_hash"]
        if output_hash != raw["output_hash"]:
            raise RuntimeError("Operator output-byte hashes differ")
        times.append(raw["run_seconds"])
        peaks.append(guard["peak_gpu_used_bytes"] / 2**30)
        evidence.extend(prefix.with_suffix(s) for s in (".json", ".log", ".guard.json"))
    ratios = [math.sqrt(a * b) / c for a, c, b in zip(*times, strict=True)]
    speed = statistics.geometric_mean(ratios)
    positive = sum(value > 1 for value in ratios)
    support = speed > 1.01 and positive >= 120
    report = {
        "status": "qmoe_unused_scratch_native_operator_screen_completed",
        "provider_sha256": build["provider_sha256"],
        "operator_matched_wall_speed_ratio": speed,
        "operator_positive_calls": positive,
        "operator_calls_per_bracket": 128,
        "operator_median_wall_seconds": [statistics.median(row) for row in times],
        "operator_sampled_peak_gpu_gib": peaks,
        "operator_output_hash": output_hash,
        "operator_support": support,
        "held_no_unchanged_retry": not support,
        "model_admission_eligible": support,
        "performance_qualified": False,
        "quality_qualified": False,
        "default_enabled": False,
        "operator_ratios_are_not_model_decode_ratios": True,
        "source_and_evidence_sha256": {str(path): sha(path) for path in [*evidence, native_path]},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "source_and_evidence_sha256"}), flush=True)


if __name__ == "__main__":
    main()
