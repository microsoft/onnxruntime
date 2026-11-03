import json
import math
from pathlib import Path
import re
import shlex
import statistics

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_fc2_rank_waves_native import guarded
from run_hc_bitcast_native import verify_static_stats


REPOS = Path("/home/sayanshaw/repos")
BUILD = REPOS / "qwen38-int4-dawn-compute-scope-reuse/build-v1"
NATIVE = BUILD.parent / "native-v1"
PARENT = REPOS / "qwen38-int4-fc2-private-a/build-v1"
VARIANT = "dawn-compute-scope-reuse-v1"


def container(path):
    path = Path(path)
    if path.is_relative_to(REPOS):
        return "/workspace/" + str(path.relative_to(REPOS))
    if path.is_relative_to(ROOT.parent):
        return "/evidence/" + str(path.relative_to(ROOT.parent))
    raise ValueError(f"Outside approved mounts: {path}")


def command_for(executable, provider, arguments, stats=None):
    shell = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
        "ORT_WEBGPU_HC_POST_NORM_PROJECTION=1 ORT_WEBGPU_HC_PROJECTION_SILU=1 "
        "ORT_WEBGPU_HC_EXACT_DENSE=1 ORT_WEBGPU_EXACT_DENSE_MID=1 "
        "ORT_WEBGPU_QMOE_COMMAND_REPLAY=0 ORT_WEBGPU_QSA_PARALLEL_MAP=1 "
        "ORT_WEBGPU_QMOE_EXTENDED_PREFILL=1 ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 "
        "ORT_WEBGPU_QMOE_FC2_PRIVATE_A=0; ")
    if stats is not None:
        shell += "export ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=" + shlex.quote(container(stats)) + "; "
    shell += "exec " + shlex.join([container(executable), container(provider), *arguments])
    return ["taskset", "-c", "0-7", "bash", "-lc", shell]


def speed_ratios(windows):
    before, candidate, after = windows
    if len(before) != 128 or len(candidate) != 128 or len(after) != 128:
        raise ValueError("Expected 128 matched operator calls per bracket")
    if any(not math.isfinite(value) or value <= 0 for row in windows for value in row):
        raise ValueError("Invalid native operator duration")
    return [math.sqrt(a * b) / c for a, c, b in zip(before, candidate, after, strict=True)]


def main():
    report_path = ROOT / f"{VARIANT}-qualification.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    manifest = json.loads((BUILD / "build-manifest.json").read_text())
    generation = json.loads((NATIVE / "generation.json").read_text())
    parent = json.loads((PARENT / "build-manifest.json").read_text())
    for records in (manifest["source_sha256"], generation["source_sha256"]):
        for path, digest in records.items():
            resolved = host_path(path) if path.startswith(("/workspace/", "/evidence/")) else Path(path)
            if sha(resolved) != digest:
                raise RuntimeError(f"Changed qualification source: {path}")
    if (not manifest["original_dawn_object_byte_identical"] or not manifest["baseline_byte_identical"]
            or manifest["baseline_provider_sha256"] != parent["provider_sha256"]):
        raise RuntimeError("Missing original Dawn and provider provenance")
    for directory, expected in ((BUILD, manifest["provider_sha256"]), (PARENT, parent["provider_sha256"])):
        if sha(directory / "libonnxruntime_providers_webgpu.so") != expected:
            raise RuntimeError("Provider changed")
    previous = json.loads((ROOT / "hc-post-norm-projection-v2-native-gpu7-20261029.qualification.json").read_text())
    fixtures = sorted((REPOS / "qwen38-int4-hc-post-norm-projection/fixtures-v1").glob("*.onnx"))
    if len(fixtures) != 36 or any(sha(p) != previous["fixture_sha256"][p.name] for p in fixtures):
        raise RuntimeError("Frozen native fixtures changed")
    evidence = [Path(__file__), GUARD, BUILD / "build-manifest.json", NATIVE / "generation.json",
                NATIVE / "native_compare", NATIVE / "native_benchmark",
                PARENT / "libonnxruntime_providers_webgpu.so", BUILD / "libonnxruntime_providers_webgpu.so"]
    fingerprints = []
    native_phases = []
    for directory, phase in ((PARENT, "original-native"), (BUILD, "candidate-native")):
        prefix = ROOT / f"{VARIANT}-{phase}-gpu0"
        stats = Path(str(prefix) + "-stats.jsonl")
        if stats.exists():
            raise FileExistsError(stats)
        command = command_for(NATIVE / "native_compare", directory / "libonnxruntime_providers_webgpu.so",
                              [container(p) for p in fixtures], stats)
        child, guard = guarded(prefix, command)
        child.check_returncode()
        text = prefix.with_suffix(".log").read_text()
        rows = re.findall(r"^PASS (\S+) changing_rounds=128 invalid_flag_rejected=1 output_fingerprint=(\d+)$",
                          text, flags=re.MULTILINE)
        if (guard["status"] != "completed" or len(rows) != 36 or "FAIL" in text
                or "PASS total_exact_elements=607143936; CPU fallback disabled" not in text):
            raise RuntimeError("Incomplete independent original/candidate native proof")
        statistics_rows = [json.loads(line) for line in stats.read_text().splitlines()]
        verify_static_stats(statistics_rows, requests=96)
        fingerprints.append(rows)
        native_phases.append({"phase": phase, "finite_bitwise_comparisons": 607143936,
                              "sampled_peak_gpu_gib": guard["peak_gpu_used_bytes"] / 2**30})
        evidence.extend([stats, *[prefix.with_suffix(s) for s in (".log", ".guard.json")]])
    if fingerprints[0] != fingerprints[1]:
        raise RuntimeError("Whole-output fingerprints differ from independently run original provider")
    native_report = {
        "status": "dawn_scope_reuse_native_cross_provider_qualified",
        "provider_sha256": manifest["provider_sha256"],
        "baseline_provider_sha256": parent["provider_sha256"],
        "phases": native_phases,
        "all_36_cross_provider_fingerprints_agree": True,
        "quality_qualified": False,
        "performance_qualified": False,
        "source_and_evidence_sha256": {str(path): sha(path) for path in evidence},
    }
    native_path = ROOT / f"{VARIANT}-native.qualification.json"
    with native_path.open("x") as stream:
        json.dump(native_report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in native_report.items()
                      if key != "source_and_evidence_sha256"}), flush=True)
    timings = []
    output_hash = None
    operator_peaks = []
    for directory, phase in ((PARENT, "operator-before"), (BUILD, "operator-candidate"),
                             (PARENT, "operator-after")):
        prefix = ROOT / f"{VARIANT}-{phase}-gpu0"
        command = command_for(NATIVE / "native_benchmark", directory / "libonnxruntime_providers_webgpu.so",
                              [container(NATIVE / "scope_batch128.onnx"),
                               container(prefix.with_suffix(".json")), "-"])
        child, guard = guarded(prefix, command)
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        if (guard["status"] != "completed" or raw["status"] != "scope_operator_benchmark_completed"
                or raw["warmups"] != 16 or raw["measured_runs"] != 128 or raw["profiling"]
                or raw["measured_major_faults"] != 0):
            raise RuntimeError("Incomplete or faulting native batch screen")
        if output_hash is None:
            output_hash = raw["output_hash"]
        if raw["output_hash"] != output_hash:
            raise RuntimeError("Independent batch output bytes differ")
        timings.append(raw["run_seconds"])
        operator_peaks.append(guard["peak_gpu_used_bytes"] / 2**30)
        evidence.extend(prefix.with_suffix(s) for s in (".json", ".log", ".guard.json"))
    ratios = speed_ratios(timings)
    speed = statistics.geometric_mean(ratios)
    positive = sum(value > 1 for value in ratios)
    support = speed > 1.01 and positive >= 120
    report = {
        "status": "dawn_scope_reuse_native_and_cpu_sensitive_batch_screen_completed",
        "provider_sha256": manifest["provider_sha256"],
        "baseline_provider_sha256": parent["provider_sha256"],
        "native_cross_provider_fingerprints_agree": True,
        "operator_matched_wall_speed_ratio": speed,
        "operator_positive_calls": positive,
        "operator_calls_per_bracket": 128,
        "operator_median_wall_seconds": [statistics.median(row) for row in timings],
        "operator_sampled_peak_gpu_gib": operator_peaks,
        "operator_output_hash": output_hash,
        "operator_support": support,
        "held_no_unchanged_retry": not support,
        "model_admission_eligible": support,
        "model_performance_qualified": False,
        "quality_qualified": False,
        "default_enabled": False,
        "operator_ratios_are_not_model_decode_ratios": True,
        "source_and_evidence_sha256": {str(path): sha(path) for path in [*evidence, native_path]},
    }
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "source_and_evidence_sha256"}), flush=True)


if __name__ == "__main__":
    main()
