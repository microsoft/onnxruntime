import json
import math
from pathlib import Path
import re
import statistics

from build_qmoe_node_replay import replace_once, sha
from run_argmax_parallel_bitwise import host_path
from run_fc2_rank_waves_native import guarded
from run_gpu_slice_model import verify_sources
import run_early_scratch_native_v1 as native_template
import run_early_scratch_operator_v2 as operator_template


ROOT = native_template.ROOT
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-invocation-flags/build-v1")
FLAG = "ORT_WEBGPU_QMOE_INVOCATION_FLAGS"
VARIANT = "qmoe-invocation-flags-v1"


def command_for(prefix, flag, replay, candidate, operator=False, profiling=False):
    if operator:
        if replay:
            raise ValueError("This operator screen uses ordinary deployed dispatch")
        command = operator_template.command_for(prefix, flag, candidate, profiling)
    else:
        command = native_template.command_for(prefix, flag, replay, candidate)
    shell = replace_once(
        command[-1], f"export {native_template.FLAG}={flag};",
        f"export {FLAG}={flag};")
    if candidate:
        shell = replace_once(
            shell, native_template.prior.mounted(
                native_template.BUILD / "libonnxruntime_providers_webgpu.so"),
            native_template.prior.mounted(BUILD / "libonnxruntime_providers_webgpu.so"))
    command[-1] = shell
    return command


def main():
    output = ROOT / f"{VARIANT}-outcome.json"
    native_path = ROOT / f"{VARIANT}-native.qualification.json"
    if output.exists() or native_path.exists():
        raise FileExistsError("Fresh invocation-flags experiment required")
    native_parent = ROOT / "fc2-private-a-v1-native-outcome.json"
    operator_parent = ROOT / "fc2-private-a-v1-operator-screen.json"
    verify_sources(native_parent, "fc2_private_a_native_finite_bitwise_and_shader_routing_qualified")
    verify_sources(operator_parent, "fc2_private_a_operator_screen_completed")
    manifest_path = BUILD / "build-manifest.json"
    build = json.loads(manifest_path.read_text())
    native_generation = native_template.prior.NATIVE / "generation.json"
    batch_generation = operator_template.OUTPUT / "generation.json"
    for path in (manifest_path, native_generation, batch_generation):
        manifest = json.loads(path.read_text())
        for source, digest in manifest["source_sha256"].items():
            resolved = host_path(source) if source.startswith(("/workspace/", "/evidence/")) else Path(source)
            if sha(resolved) != digest:
                raise RuntimeError(f"Changed qualification source: {source}")
    if (not build["baseline_byte_identical"] or not build["original_qmoe_object_byte_identical"]
            or not build["no_persistent_value_cache"]
            or sha(BUILD / "libonnxruntime_providers_webgpu.so") != build["provider_sha256"]
            or sha(native_template.prior.PARENT / "libonnxruntime_providers_webgpu.so")
            != build["baseline_provider_sha256"]):
        raise RuntimeError("Missing supported original/provider/source proof")
    evidence = [
        Path(__file__), Path(native_template.__file__), Path(operator_template.__file__),
        Path(native_template.prior.__file__), native_template.GUARD,
        native_parent, operator_parent, manifest_path,
        BUILD / "libonnxruntime_providers_webgpu.so",
        native_template.prior.PARENT / "libonnxruntime_providers_webgpu.so",
        native_generation, native_template.prior.NATIVE / "native_compare",
        native_template.prior.NATIVE / "native_compare.cpp",
        batch_generation, native_template.BATCH / "native_benchmark",
        operator_template.FIXTURE,
    ]
    fingerprints = None
    phases = []
    for flag, replay, candidate, phase in (
            (0, 0, False, "original"), (0, 0, True, "default-off"),
            (1, 0, True, "selected"), (1, 1, True, "replay"),
            (2, 0, True, "invalid")):
        prefix = ROOT / f"{VARIANT}-native-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(prefix, flag, replay, candidate))
        text = prefix.with_suffix(".log").read_text()
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".log", ".guard.json"))
        if flag == 2:
            if child.returncode != 1 or f"{FLAG} must be 0 or 1" not in text:
                raise RuntimeError("Invalid invocation selector did not reject explicitly")
            phases.append({"phase": phase, "invalid_selector_rejected": True})
            continue
        child.check_returncode()
        rows = re.findall(
            r"^PASS changing_inputs_and_routes (\S+) rounds=32 invalid_flag_rejected=1 output_fingerprint=(\d+)$",
            text, flags=re.MULTILINE)
        if (guard["status"] != "completed" or len(rows) != 23 or "FAIL" in text
                or "PASS total_exact_elements=9748480; CPU fallback disabled" not in text):
            raise RuntimeError("Incomplete changing-input/router/selector finite proof")
        if fingerprints is None:
            fingerprints = rows
        if rows != fingerprints:
            raise RuntimeError("Independent original/default/selected/replay fingerprints differ")
        phases.append({
            "phase": phase, "flag": flag, "command_replay": replay,
            "finite_bitwise_elements": 9748480, "fixtures": 23,
            "all_output_fingerprints_agree": True,
            "sampled_peak_gpu_gib": guard["peak_gpu_used_bytes"] / 2**30,
        })
        print(json.dumps(phases[-1]), flush=True)
    native = {
        "status": "qmoe_invocation_flags_native_cross_provider_qualified",
        "provider_sha256": build["provider_sha256"], "phases": phases,
        "performance_qualified": False, "quality_qualified": False,
        "source_and_evidence_sha256": {str(path): sha(path) for path in evidence},
    }
    with native_path.open("x") as stream:
        json.dump(native, stream, indent=2)
        stream.write("\n")
    evidence.append(native_path)
    routing = []
    output_hash = None
    for flag, candidate, phase in ((0, False, "control"), (1, True, "candidate")):
        prefix = ROOT / f"{VARIANT}-routing-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(prefix, flag, 0, candidate, True, True))
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        native_template.validate_operator(raw, guard, True)
        profile = host_path(raw["profile_path"])
        counts = native_template.validate_routing(json.loads(profile.read_text()))
        if output_hash is None:
            output_hash = raw["output_hash"]
        if raw["output_hash"] != output_hash:
            raise RuntimeError("Routing output-byte hashes differ")
        routing.append({"phase": phase, "program_counts": counts})
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".json", ".log", ".guard.json"))
        evidence.append(profile)
    if routing[0]["program_counts"] != routing[1]["program_counts"]:
        raise RuntimeError("Control/candidate deployed routing differs")
    times, peaks = [], []
    for flag, candidate, phase in ((0, False, "before"), (1, True, "candidate"), (0, False, "after")):
        prefix = ROOT / f"{VARIANT}-operator-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(prefix, flag, 0, candidate, True))
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        native_template.validate_operator(raw, guard, False)
        if raw["output_hash"] != output_hash:
            raise RuntimeError("Timing output-byte hashes differ")
        times.append(raw["run_seconds"])
        peaks.append(guard["peak_gpu_used_bytes"] / 2**30)
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".json", ".log", ".guard.json"))
    ratios = [math.sqrt(a * b) / c for a, c, b in zip(*times, strict=True)]
    speed = statistics.geometric_mean(ratios)
    positive = sum(value > 1 for value in ratios)
    support = speed > 1.01 and positive >= 120
    report = {
        "status": "qmoe_invocation_flags_native_and_batch_screen_completed",
        "provider_sha256": build["provider_sha256"],
        "baseline_provider_sha256": build["baseline_provider_sha256"],
        "actual_deployed_gate_routing": routing,
        "qmoe_nodes_per_operator_call": 32,
        "batch_reuses_weights_inputs_and_routes_not_model_latency": True,
        "operator_matched_wall_speed_ratio": speed,
        "operator_positive_calls": positive,
        "operator_calls_per_bracket": 128,
        "operator_median_wall_seconds": [statistics.median(row) for row in times],
        "operator_sampled_peak_gpu_gib": peaks, "operator_output_hash": output_hash,
        "operator_support": support, "model_admission_eligible": support,
        "held_no_unchanged_retry": not support,
        "performance_qualified": False, "quality_qualified": False,
        "default_enabled": False, "no_persistent_value_cache": True,
        "original_provider_before_after_controls": True,
        "operator_ratios_are_not_model_decode_ratios": True,
        "source_and_evidence_sha256": {str(path): sha(path) for path in evidence},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "source_and_evidence_sha256"}), flush=True)


if __name__ == "__main__":
    main()
