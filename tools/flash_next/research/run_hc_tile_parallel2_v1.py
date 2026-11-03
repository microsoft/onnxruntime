import json
import math
from pathlib import Path
import re
import statistics

from build_qmoe_node_replay import sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_fc2_rank_waves_native import guarded
from run_hc_bitcast_native import verify_static_stats
from run_dawn_scope_qualification_v1 import NATIVE, PARENT, REPOS, command_for as base_command, container
from run_hc_small_cta_operator_v1 import BENCHMARK, FIXTURE


BUILD = REPOS / "qwen38-int4-hc-tile-parallel2/build-v1"
FLAG = "ORT_WEBGPU_HC_TILE_PARALLEL2"
NAME = "HCPostNormProjectionTileParallel2F16"
VARIANT = "hc-tile-parallel2-v1"


def command_for(executable, prefix, flag, candidate, arguments, stats=None):
    if flag not in (0, 1, 2):
        raise ValueError("Invalid owned-pair selector")
    provider = (BUILD if candidate else PARENT) / "libonnxruntime_providers_webgpu.so"
    command = base_command(executable, provider, arguments, stats)
    command[-1] = f"export {FLAG}={flag}; " + command[-1]
    return command


def measured_gpu(events, selected):
    expected = NAME if selected else "HCPostNormProjectionF16"
    kernels = [event for event in events if event.get("cat") == "Kernel"]
    if len(kernels) != 144 or any(
            not event.get("name", "").endswith("&" + expected)
            or event.get("pid") != -1 or event.get("tid") != -1 or event.get("ph") != "X"
            or not math.isfinite(event.get("dur", 0)) or event["dur"] <= 0
            or (":512,1,1:" if selected else ":256,1,1:") not in event.get("args", {}).get("cache_key", "") for event in kernels):
        raise RuntimeError("Wrong actual HC program/count/workgroup")
    return [event["dur"] for event in kernels[16:]]


def main():
    output = ROOT / f"{VARIANT}-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    build = json.loads((BUILD / "build-manifest.json").read_text())
    generation = json.loads((NATIVE / "generation.json").read_text())
    for records in (build["source_sha256"], generation["source_sha256"]):
        for path, digest in records.items():
            path = host_path(path) if path.startswith(("/workspace/", "/evidence/")) else Path(path)
            if sha(path) != digest:
                raise RuntimeError(f"Changed qualification source: {path}")
    if (not build["baseline_byte_identical"] or not build["original_hc_object_byte_identical"]
            or sha(BUILD / "libonnxruntime_providers_webgpu.so") != build["provider_sha256"]
            or sha(PARENT / "libonnxruntime_providers_webgpu.so") != build["baseline_provider_sha256"]):
        raise RuntimeError("Missing supported original HC/provider provenance")
    previous_path = ROOT / "hc-post-norm-projection-v2-native-gpu7-20261029.qualification.json"
    previous = json.loads(previous_path.read_text())
    fixtures = sorted((REPOS / "qwen38-int4-hc-post-norm-projection/fixtures-v1").glob("*.onnx"))
    if (len(fixtures) != 36 or any(sha(path) != previous["fixture_sha256"][path.name] for path in fixtures)
            or sha(FIXTURE) != previous["fixture_sha256"][FIXTURE.name]):
        raise RuntimeError("Frozen native/operator fixtures changed")
    evidence = [Path(__file__), ROOT / "prepare_hc_tile_parallel2.py",
                ROOT / "run_hc_owned_pair34_v1.py", GUARD, previous_path, BUILD / "build-manifest.json",
                BUILD / "libonnxruntime_providers_webgpu.so", PARENT / "libonnxruntime_providers_webgpu.so",
                NATIVE / "generation.json", NATIVE / "native_compare", BENCHMARK,
                ROOT / "hc_small_cta_operator_benchmark.cpp", ROOT / "run_dawn_scope_qualification_v1.py"]
    fingerprints, phases = None, []
    for flag, candidate, phase in ((0, False, "original"), (0, True, "default-off"),
                                   (1, True, "selected"), (2, True, "invalid")):
        prefix = ROOT / f"{VARIANT}-native-{phase}-gpu0"
        stats = Path(str(prefix) + "-stats.jsonl")
        if stats.exists():
            raise FileExistsError(stats)
        paths = fixtures[:1] if flag == 2 else fixtures
        command = command_for(NATIVE / "native_compare", prefix, flag, candidate,
                              [container(path) for path in paths], stats)
        child, guard = guarded(prefix, command)
        text = prefix.with_suffix(".log").read_text()
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".log", ".guard.json"))
        if flag == 2:
            if child.returncode != 1 or f"{FLAG} must be 0 or 1" not in text:
                raise RuntimeError("Invalid process-static owned-pair selector did not reject")
            phases.append({"phase": phase, "invalid_selector_rejected": True})
            continue
        child.check_returncode()
        rows = re.findall(r"^PASS (\S+) changing_rounds=128 invalid_flag_rejected=1 output_fingerprint=(\d+)$",
                          text, flags=re.MULTILINE)
        if (guard["status"] != "completed" or len(rows) != 36 or "FAIL" in text
                or "PASS total_exact_elements=607143936; CPU fallback disabled" not in text):
            raise RuntimeError("Incomplete independent HC native proof")
        verify_static_stats([json.loads(line) for line in stats.read_text().splitlines()], requests=96)
        if fingerprints is None:
            fingerprints = rows
        if rows != fingerprints:
            raise RuntimeError("Independent original/default/selected output bytes differ")
        phases.append({"phase": phase, "finite_bitwise_comparisons": 607143936,
                       "fixtures": 36, "all_output_fingerprints_agree": True,
                       "sampled_peak_gpu_gib": guard["peak_gpu_used_bytes"] / 2**30})
        evidence.append(stats)
        print(json.dumps(phases[-1]), flush=True)
    native = {"status": "hc_tile_parallel2_native_cross_provider_qualified",
              "provider_sha256": build["provider_sha256"], "phases": phases,
              "performance_qualified": False, "quality_qualified": False,
              "source_and_evidence_sha256": {str(path): sha(path) for path in evidence}}
    with (ROOT / f"{VARIANT}-native.qualification.json").open("x") as stream:
        json.dump(native, stream, indent=2)
        stream.write("\n")
    wall, profiles, peaks = [], [], []
    output_hash = None
    for flag, candidate, phase, profiling in (
            (0, False, "before", False), (1, True, "candidate", False), (0, False, "after", False),
            (0, False, "profile-before", True), (1, True, "profile-candidate", True),
            (0, False, "profile-after", True)):
        prefix = ROOT / f"{VARIANT}-operator-{phase}-gpu0"
        arguments = [container(FIXTURE), container(prefix.with_suffix(".json")),
                     container(Path(str(prefix) + "-ort-profile")) if profiling else "-"]
        child, guard = guarded(prefix, command_for(BENCHMARK, prefix, flag, candidate, arguments))
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        if (guard["status"] != "completed" or raw["status"] != "hc_operator_benchmark_completed"
                or raw["warmups"] != 16 or raw["measured_runs"] != 128
                or len(raw["run_seconds"]) != 128 or raw["profiling"] != profiling
                or raw["measured_major_faults"] != 0
                or any(not math.isfinite(value) or value <= 0 for value in raw["run_seconds"])):
            raise RuntimeError("Wrong HC operator scope, faults or completion")
        if output_hash is None:
            output_hash = raw["output_hash"]
        if raw["output_hash"] != output_hash:
            raise RuntimeError("All-four-output byte hash differs between operator brackets")
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".json", ".log", ".guard.json"))
        peaks.append(guard["peak_gpu_used_bytes"] / 2**30)
        if profiling:
            path = host_path(raw["profile_path"])
            durations = measured_gpu(json.loads(path.read_text()), bool(flag))
            profiles.append({"phase": phase, "median_gpu_us": statistics.median(durations), "gpu_us": durations})
            evidence.append(path)
        else:
            wall.append(raw["run_seconds"])
    gpu_ratios = [math.sqrt(a * b) / c
                  for a, c, b in zip(*(profile["gpu_us"] for profile in profiles), strict=True)]
    wall_ratios = [math.sqrt(a * b) / c for a, c, b in zip(*wall, strict=True)]
    speed = statistics.geometric_mean(gpu_ratios)
    positives = sum(value > 1 for value in gpu_ratios)
    support = speed > 1.01 and positives >= 120
    report = {
        "status": "hc_tile_parallel2_native_operator_screen_completed",
        "provider_sha256": build["provider_sha256"],
        "native_phases": phases,
        "profiles": profiles,
        "operator_matched_gpu_speed_ratio": speed,
        "operator_positive_gpu_calls": positives,
        "operator_matched_wall_speed_ratio": statistics.geometric_mean(wall_ratios),
        "operator_median_wall_seconds": [statistics.median(values) for values in wall],
        "operator_sampled_peak_gpu_gib": peaks,
        "operator_output_hash": output_hash,
        "operator_support": support,
        "held_no_unchanged_retry": not support,
        "model_admission_eligible": support,
        "performance_qualified": False,
        "quality_qualified": False,
        "default_enabled": False,
        "operator_ratios_are_not_model_decode_ratios": True,
        "source_and_evidence_sha256": {str(path): sha(path) for path in evidence},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("source_and_evidence_sha256", "profiles")}), flush=True)


if __name__ == "__main__":
    main()
