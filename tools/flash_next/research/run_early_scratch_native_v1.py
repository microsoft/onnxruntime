import collections
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
import run_unused_scratch_native_v2 as prior


BUILD = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-early-scratch/build-v1")
FLAG = "ORT_WEBGPU_QMOE_EARLY_SCRATCH"
VARIANT = "qmoe-early-scratch-v1"
BATCH = BUILD.parent / "native-v1"


def command_for(prefix, flag, replay, candidate, operator=False, profiling=False):
    if profiling and not operator:
        raise ValueError("Profiling is supported only for operator diagnostics")
    command = prior.command_for(prefix, flag, replay, False, operator)
    shell = replace_once(command[-1], f"export {prior.FLAG}={flag};", f"export {FLAG}={flag};")
    if candidate:
        shell = replace_once(shell, prior.mounted(prior.PARENT / "libonnxruntime_providers_webgpu.so"),
                             prior.mounted(BUILD / "libonnxruntime_providers_webgpu.so"))
    if operator:
        shell = replace_once(shell, prior.mounted(prior.PARENT / "native_benchmark"),
                             prior.mounted(BATCH / "native_benchmark"))
        shell = replace_once(shell, "/workspace/qwen38-int4-qmoe-node-replay/fixtures/expert_layer0_rows1.onnx",
                             prior.mounted(BATCH / "qmoe_batch32.onnx"))
    if "ORT_WEBGPU_QMOE_SUBGROUP_GATE=" in shell:
        raise RuntimeError("Unexpected inherited subgroup selector")
    shell = "export ORT_WEBGPU_QMOE_SUBGROUP_GATE=1; " + shell
    if "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1" not in shell:
        raise RuntimeError("Missing deployed hierarchical gate selector")
    if profiling:
        if not shell.endswith(" -"):
            raise RuntimeError("Unexpected operator profile argument")
        shell = shell[:-2] + f" /evidence/webgpu-perf-oct20/{prefix.name}-ort-profile"
    command[-1] = shell
    return command


def validate_operator(raw, guard, profiling):
    values = raw["run_seconds"]
    if (guard["status"] != "completed" or raw["status"] != "qmoe_operator_benchmark_completed"
            or raw["warmups"] != 16 or raw["measured_runs"] != 128
            or raw["profiling"] != profiling or raw["measured_major_faults"] != 0
            or len(values) != 128
            or any(not math.isfinite(value) or value <= 0 for value in values)):
        raise RuntimeError("Incomplete, invalid or faulting native operator bracket")


def validate_routing(events):
    kernels = [event for event in events if event.get("cat") == "Kernel"]
    expected = {"QmoeHierarchicalGate1Token", "QmoeFC1SwiGLUBitcastF16", "QmoeFC2FinalMixBitcastF16"}
    counts = collections.Counter()
    for event in kernels:
        name = event["name"].split("&")[-1]
        if (event.get("pid") != -1 or event.get("tid") != -1 or event.get("ph") != "X"
                or not math.isfinite(event.get("dur", 0)) or event["dur"] <= 0):
            raise RuntimeError("Invalid actual GPU event")
        counts[name] += 1
    if (any(counts[name] != 144 * 32 for name in expected)
            or set(counts) != expected | {"Concat"}
            or counts["Concat"] % 144 != 0 or not 1 <= counts["Concat"] // 144 <= 32):
        raise RuntimeError(f"Wrong deployed gate/fused projection routing: {counts}")
    for event in kernels:
        name = event["name"].split("&")[-1]
        threads = {"QmoeHierarchicalGate1Token": 512, "QmoeFC1SwiGLUBitcastF16": 128,
                   "QmoeFC2FinalMixBitcastF16": 320}.get(name)
        if threads and f":{threads},1,1:" not in event.get("args", {}).get("cache_key", ""):
            raise RuntimeError(f"Wrong deployed workgroup size for {name}")
    return dict(counts)


def main():
    output = ROOT / f"{VARIANT}-native-operator-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    native_parent = ROOT / "fc2-private-a-v1-native-outcome.json"
    operator_parent = ROOT / "fc2-private-a-v1-operator-screen.json"
    verify_sources(native_parent, "fc2_private_a_native_finite_bitwise_and_shader_routing_qualified")
    verify_sources(operator_parent, "fc2_private_a_operator_screen_completed")
    build = json.loads((BUILD / "build-manifest.json").read_text())
    generation = json.loads((prior.NATIVE / "generation.json").read_text())
    batch = json.loads((BATCH / "generation.json").read_text())
    for records in (build["source_sha256"], generation["source_sha256"], batch["source_sha256"]):
        for path, digest in records.items():
            path = host_path(path) if path.startswith(("/workspace/", "/evidence/")) else Path(path)
            if sha(path) != digest:
                raise RuntimeError(f"Changed qualification source: {path}")
    if (not build["baseline_byte_identical"] or not build["original_qmoe_object_byte_identical"]
            or sha(BUILD / "libonnxruntime_providers_webgpu.so") != build["provider_sha256"]
            or sha(prior.PARENT / "libonnxruntime_providers_webgpu.so") != build["baseline_provider_sha256"]):
        raise RuntimeError("Missing original object/provider provenance")
    evidence = [Path(__file__), Path(prior.__file__), GUARD, native_parent, operator_parent,
                BUILD / "build-manifest.json", BUILD / "libonnxruntime_providers_webgpu.so",
                prior.NATIVE / "generation.json", prior.NATIVE / "native_compare",
                prior.PARENT / "libonnxruntime_providers_webgpu.so",
                BATCH / "generation.json", BATCH / "native_benchmark", BATCH / "qmoe_batch32.onnx"]
    fingerprints = None
    phases = []
    for flag, replay, candidate, phase in (
            (0, 0, False, "original"), (0, 0, True, "default-off"),
            (1, 0, True, "selected"), (1, 1, True, "replay-fallback"),
            (2, 0, True, "invalid")):
        prefix = ROOT / f"{VARIANT}-native-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(prefix, flag, replay, candidate))
        text = prefix.with_suffix(".log").read_text()
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".log", ".guard.json"))
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
        print(json.dumps(phases[-1]), flush=True)
    native = {"status": "qmoe_early_scratch_native_cross_provider_qualified",
              "provider_sha256": build["provider_sha256"], "phases": phases,
              "performance_qualified": False, "quality_qualified": False,
              "source_and_evidence_sha256": {str(path): sha(path) for path in evidence}}
    with (ROOT / f"{VARIANT}-native.qualification.json").open("x") as stream:
        json.dump(native, stream, indent=2)
        stream.write("\n")
    output_hash = None
    routing = []
    for flag, candidate, phase in ((0, False, "control"), (1, True, "candidate")):
        prefix = ROOT / f"{VARIANT}-routing-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(prefix, flag, 0, candidate, True, True))
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        validate_operator(raw, guard, True)
        profile = host_path(raw["profile_path"])
        counts = validate_routing(json.loads(profile.read_text()))
        if output_hash is None:
            output_hash = raw["output_hash"]
        if raw["output_hash"] != output_hash:
            raise RuntimeError("Routing diagnostic output hashes differ")
        routing.append({"phase": phase, "program_counts": counts})
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".json", ".log", ".guard.json"))
        evidence.append(profile)
    if routing[0]["program_counts"] != routing[1]["program_counts"]:
        raise RuntimeError("Control/candidate batch dispatch counts differ")
    times, peaks = [], []
    for flag, candidate, phase in ((0, False, "before"), (1, True, "candidate"), (0, False, "after")):
        prefix = ROOT / f"{VARIANT}-operator-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(prefix, flag, 0, candidate, True))
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        validate_operator(raw, guard, False)
        if raw["output_hash"] != output_hash:
            raise RuntimeError("Operator output-byte hashes differ")
        times.append(raw["run_seconds"])
        peaks.append(guard["peak_gpu_used_bytes"] / 2**30)
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".json", ".log", ".guard.json"))
    ratios = [math.sqrt(a * b) / c for a, c, b in zip(*times, strict=True)]
    speed = statistics.geometric_mean(ratios)
    positive = sum(value > 1 for value in ratios)
    support = speed > 1.01 and positive >= 120
    report = {
        "status": "qmoe_early_scratch_native_operator_screen_completed",
        "provider_sha256": build["provider_sha256"],
        "baseline_provider_sha256": build["baseline_provider_sha256"],
        "native_phases": phases,
        "actual_deployed_gate_routing": routing,
        "qmoe_nodes_per_operator_call": 32,
        "batch_reuses_weights_inputs_and_routes_not_model_latency": True,
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
