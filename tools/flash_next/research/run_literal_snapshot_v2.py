import collections
import json
import math
from pathlib import Path
import re
import statistics

from attribute_cpu_allocations_v2 import sha
from build_qmoe_node_replay import replace_once
from run_argmax_parallel_bitwise import host_path
from run_fc2_rank_waves_native import guarded
from run_gpu_slice_model import verify_sources
import run_early_scratch_native_v1 as native_template
import run_early_scratch_operator_v2 as operator_template


ROOT = native_template.ROOT
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-webgpu-literal-snapshot/build-v1")
NATIVE = BUILD.parent / "native-v1"
PARENT = native_template.prior.PARENT
FLAG = "ORT_WEBGPU_LITERAL_SNAPSHOT"
VARIANT = "webgpu-literal-snapshot-v2"
GOLDEN = ROOT / f"{VARIANT}-native-golden.bin"


def mounted(path):
    return native_template.prior.mounted(path)


def command_for(prefix, flag, replay, candidate, operator=False, profiling=False,
                write_reference=False, legacy_invalid=False):
    if operator:
        command = operator_template.command_for(prefix, flag, False, profiling)
        command[-1] = replace_once(
            command[-1], "ORT_WEBGPU_QMOE_COMMAND_REPLAY=0",
            f"ORT_WEBGPU_QMOE_COMMAND_REPLAY={replay}")
    else:
        command = native_template.command_for(prefix, flag, replay, False)
    shell = replace_once(
        command[-1], f"export {native_template.FLAG}={flag};", f"export {FLAG}={flag};")
    provider = mounted(PARENT / "libonnxruntime_providers_webgpu.so")
    if candidate:
        replacement = mounted(BUILD / "libonnxruntime_providers_webgpu.so")
        shell = replace_once(shell, provider, replacement)
        provider = replacement
    if not operator:
        old = mounted(native_template.prior.NATIVE / "native_compare")
        mode = "write" if write_reference else "compare"
        shell = replace_once(
            shell, f"{old} {provider}",
            f"{mounted(NATIVE / 'native_fixed')} {provider} {mode} "
            f"/evidence/webgpu-perf-oct20/{GOLDEN.name}")
        if ("ORT_WEBGPU_QMOE_FC2_FINAL_MIX=" in shell
                or "ORT_WEBGPU_MATMUL_GEMV_N96=" in shell):
            raise RuntimeError("Unexpected inherited mutable-native startup setting")
        fc2 = 2 if legacy_invalid else (0 if write_reference else 1)
        shell = (f"export ORT_WEBGPU_QMOE_FC2_FINAL_MIX={fc2}; "
                 "export ORT_WEBGPU_MATMUL_GEMV_N96=1; " + shell)
    if operator:
        if "ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE=" in shell:
            raise RuntimeError("Unexpected inherited stats output")
        shell = (
            "export ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE="
            f"/evidence/webgpu-perf-oct20/{prefix.name}-stats.jsonl; " + shell)
    command[-1] = shell
    return command


def replay_counters(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (len(rows) != 32 or len({row["kernel_identity"] for row in rows}) != 32
            or any(row["requests"] != 144 or row["hits"] < 128
                   or row["captures"] + row["hits"] != row["requests"]
                   or row["shape_invalidations"] != 0
                   or row["capacity_fallbacks"] != 0 for row in rows)):
        raise RuntimeError("Replay-enabled timing lacks matched high-hit 32-node evidence")
    return rows


def main():
    output = ROOT / f"{VARIANT}-outcome.json"
    native_path = ROOT / f"{VARIANT}-native.qualification.json"
    if output.exists() or native_path.exists() or GOLDEN.exists():
        raise FileExistsError("Fresh literal-snapshot evidence required")
    native_parent = ROOT / "fc2-private-a-v1-native-outcome.json"
    verify_sources(native_parent, "fc2_private_a_native_finite_bitwise_and_shader_routing_qualified")
    manifest_path = BUILD / "build-manifest.json"
    build = json.loads(manifest_path.read_text())
    generation_path = NATIVE / "generation.json"
    generation = json.loads(generation_path.read_text())
    batch_generation = operator_template.OUTPUT / "generation.json"
    for metadata in (build, generation, json.loads(batch_generation.read_text())):
        for path, digest in metadata["source_sha256"].items():
            resolved = host_path(path) if path.startswith(("/workspace/", "/evidence/")) else Path(path)
            if resolved.is_symlink() and (target := resolved.readlink()).is_absolute():
                if not str(target).startswith(("/workspace/", "/evidence/")):
                    raise RuntimeError(f"Unmapped absolute SDK input symlink: {target}")
                resolved = host_path(str(target))
            if sha(resolved) != digest:
                raise RuntimeError(f"Changed qualification source: {path}")
    if (not build["all_three_original_objects_byte_identical"]
            or not build["baseline_provider_byte_identical"]
            or sha(BUILD / "libonnxruntime_providers_webgpu.so") != build["provider_sha256"]
            or sha(PARENT / "libonnxruntime_providers_webgpu.so") != build["baseline_provider_sha256"]):
        raise RuntimeError("Missing supported carrier/object proofs")
    evidence = [
        Path(__file__), ROOT / "literal-snapshot-launch-v2-generation.json",
        Path(native_template.__file__), Path(operator_template.__file__),
        Path(native_template.prior.__file__), native_template.GUARD, native_parent,
        manifest_path, generation_path, NATIVE / "native_fixed",
        BUILD / "libonnxruntime_providers_webgpu.so",
        PARENT / "libonnxruntime_providers_webgpu.so", batch_generation,
        native_template.BATCH / "native_benchmark", operator_template.FIXTURE,
    ]
    fingerprints = None
    phases = []
    for flag, replay, candidate, phase, writer, legacy_invalid in (
            (0, 0, False, "original-unfused-reference", True, False),
            (0, 0, False, "original-fused", False, False),
            (0, 0, True, "default-off", False, False),
            (1, 0, True, "selected", False, False),
            (1, 1, True, "selected-replay", False, False),
            (2, 0, True, "invalid-private", False, False),
            (1, 0, True, "invalid-legacy-fresh-process", False, True)):
        prefix = ROOT / f"{VARIANT}-native-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(
            prefix, flag, replay, candidate, write_reference=writer,
            legacy_invalid=legacy_invalid))
        text = prefix.with_suffix(".log").read_text()
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".log", ".guard.json"))
        if flag == 2 or legacy_invalid:
            expected = FLAG if flag == 2 else "ORT_WEBGPU_QMOE_FC2_FINAL_MIX"
            if child.returncode != 1 or f"{expected} must be 0 or 1" not in text:
                raise RuntimeError("Fresh invalid startup setting did not reject explicitly")
            phases.append({"phase": phase, "invalid_selector_rejected": True})
            continue
        child.check_returncode()
        rows = re.findall(
            r"^PASS changing_inputs_and_routes (\S+) rounds=32 fixed_startup_flags=1 output_fingerprint=(\d+)$",
            text, flags=re.MULTILINE)
        marker = "finite" if writer else "exact"
        if (guard["status"] != "completed" or len(rows) != 23 or "FAIL" in text
                or f"PASS total_{marker}_elements=9748480; CPU fallback disabled" not in text):
            raise RuntimeError("Incomplete fixed-startup finite/golden byte qualification")
        if fingerprints is None:
            fingerprints = rows
        if rows != fingerprints:
            raise RuntimeError("Independent unfused/default/selected/replay fingerprints differ")
        phases.append({
            "phase": phase, "finite_elements": 9748480,
            "bitwise_elements_compared_to_independent_golden": 0 if writer else 9748480,
            "fixtures": 23, "rank_shape_and_count_checked": True,
            "all_output_fingerprints_agree": True,
            "sampled_peak_gpu_gib": guard["peak_gpu_used_bytes"] / 2**30,
        })
        print(json.dumps(phases[-1]), flush=True)
    evidence.append(GOLDEN)
    native = {
        "status": "literal_snapshot_fresh_process_native_golden_qualified",
        "provider_sha256": build["provider_sha256"], "phases": phases,
        "fresh_process_required_for_startup_setting_changes": True,
        "performance_qualified": False, "quality_qualified": False,
        "source_and_evidence_sha256": {str(path): sha(path) for path in evidence},
    }
    with native_path.open("x") as stream:
        json.dump(native, stream, indent=2)
        stream.write("\n")
    evidence.append(native_path)
    output_hash = None
    routing = []
    for flag, candidate, phase in ((0, False, "control"), (1, True, "candidate")):
        prefix = ROOT / f"{VARIANT}-routing-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(prefix, flag, 1, candidate, True, True))
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        native_template.validate_operator(raw, guard, True)
        profile = host_path(raw["profile_path"])
        counts = native_template.validate_routing(json.loads(profile.read_text()))
        if output_hash is None:
            output_hash = raw["output_hash"]
        if raw["output_hash"] != output_hash:
            raise RuntimeError("Routing output bytes differ")
        routing.append({"phase": phase, "program_counts": counts})
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".json", ".log", ".guard.json"))
        evidence.append(profile)
    if routing[0]["program_counts"] != routing[1]["program_counts"]:
        raise RuntimeError("Control/candidate deployed GPU routing differs")
    times, peaks, counters = [], [], []
    for flag, candidate, phase in ((0, False, "before"), (1, True, "candidate"), (0, False, "after")):
        prefix = ROOT / f"{VARIANT}-operator-{phase}-gpu0"
        stats = ROOT / f"{prefix.name}-stats.jsonl"
        if stats.exists():
            raise FileExistsError(stats)
        child, guard = guarded(prefix, command_for(prefix, flag, 1, candidate, True))
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        native_template.validate_operator(raw, guard, False)
        if raw["output_hash"] != output_hash:
            raise RuntimeError("Timing output bytes differ")
        counts = replay_counters(stats)
        counters.append(counts)
        times.append(raw["run_seconds"])
        peaks.append(guard["peak_gpu_used_bytes"] / 2**30)
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".json", ".log", ".guard.json"))
        evidence.append(stats)
    signatures = [
        collections.Counter((row["requests"], row["hits"], row["captures"],
                             row["evictions"]) for row in rows) for rows in counters]
    if not signatures[0] == signatures[1] == signatures[2]:
        raise RuntimeError("Replay cache behavior differs between timing brackets")
    ratios = [math.sqrt(a * b) / c for a, c, b in zip(*times, strict=True)]
    speed = statistics.geometric_mean(ratios)
    positive = sum(value > 1 for value in ratios)
    support = speed > 1.01 and positive >= 120
    report = {
        "status": "literal_snapshot_native_and_replay_enabled_batch_screen_completed",
        "provider_sha256": build["provider_sha256"],
        "baseline_provider_sha256": build["baseline_provider_sha256"],
        "actual_deployed_gate_routing": routing,
        "qmoe_nodes_per_operator_call": 32, "command_replay_enabled": True,
        "per_bracket_replay_counters": counters,
        "batch_reuses_weights_inputs_and_routes_not_model_latency": True,
        "operator_matched_wall_speed_ratio": speed,
        "operator_positive_calls": positive,
        "operator_calls_per_bracket": 128,
        "operator_median_wall_seconds": [statistics.median(row) for row in times],
        "operator_sampled_peak_gpu_gib": peaks, "operator_output_hash": output_hash,
        "operator_support": support, "model_admission_eligible": support,
        "held_no_unchanged_retry": not support,
        "performance_qualified": False, "quality_qualified": False,
        "default_enabled": False, "fresh_process_required": True,
        "original_provider_before_after_controls": True,
        "operator_ratios_are_not_model_decode_ratios": True,
        "source_and_evidence_sha256": {str(path): sha(path) for path in evidence},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("source_and_evidence_sha256", "per_bracket_replay_counters")}),
          flush=True)


if __name__ == "__main__":
    main()
