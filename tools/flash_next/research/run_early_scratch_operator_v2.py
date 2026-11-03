import json
import math
from pathlib import Path
import statistics

from build_qmoe_node_replay import replace_once, sha
from prepare_early_scratch_batch_v2 import FIXTURE, OUTPUT
from run_argmax_parallel_bitwise import host_path
from run_fc2_rank_waves_native import guarded
from run_gpu_slice_model import verify_sources
import run_early_scratch_native_v1 as prior


ROOT = prior.ROOT
VARIANT = "qmoe-early-scratch-v2"


def command_for(prefix, flag, candidate, profiling=False):
    command = prior.command_for(prefix, flag, 0, candidate, True, profiling)
    command[-1] = replace_once(command[-1], prior.prior.mounted(prior.BATCH / "qmoe_batch32.onnx"),
                               prior.prior.mounted(FIXTURE))
    return command


def main():
    output = ROOT / f"{VARIANT}-operator-outcome.json"
    if output.exists():
        raise FileExistsError(output)
    native_path = ROOT / "qmoe-early-scratch-v1-native.qualification.json"
    native = verify_sources(native_path, "qmoe_early_scratch_native_cross_provider_qualified")
    build = json.loads((prior.BUILD / "build-manifest.json").read_text())
    generation = json.loads((OUTPUT / "generation.json").read_text())
    if (native["provider_sha256"] != build["provider_sha256"]
            or len(native["phases"]) != 5
            or any(phase.get("finite_bitwise_elements") != 9748480
                   or not phase.get("all_output_fingerprints_agree") for phase in native["phases"][:4])
            or not native["phases"][-1].get("invalid_selector_rejected")):
        raise RuntimeError("Missing complete native qualification")
    for records in (build["source_sha256"], generation["source_sha256"]):
        for path, digest in records.items():
            path = host_path(path) if path.startswith(("/workspace/", "/evidence/")) else Path(path)
            if sha(path) != digest:
                raise RuntimeError(f"Changed qualification source: {path}")
    if (not build["baseline_byte_identical"] or not build["original_qmoe_object_byte_identical"]
            or sha(prior.BUILD / "libonnxruntime_providers_webgpu.so") != build["provider_sha256"]
            or sha(prior.prior.PARENT / "libonnxruntime_providers_webgpu.so") != build["baseline_provider_sha256"]):
        raise RuntimeError("Missing exact provider provenance")
    evidence = [Path(__file__), Path(prior.__file__), Path(prior.prior.__file__), prior.GUARD,
                native_path, prior.BUILD / "build-manifest.json", OUTPUT / "generation.json",
                prior.BUILD / "libonnxruntime_providers_webgpu.so",
                prior.prior.PARENT / "libonnxruntime_providers_webgpu.so",
                prior.BATCH / "native_benchmark", FIXTURE]
    output_hash = None
    routing = []
    for flag, candidate, phase in ((0, False, "control"), (1, True, "candidate")):
        prefix = ROOT / f"{VARIANT}-routing-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(prefix, flag, candidate, True))
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        prior.validate_operator(raw, guard, True)
        profile = host_path(raw["profile_path"])
        counts = prior.validate_routing(json.loads(profile.read_text()))
        if output_hash is None:
            output_hash = raw["output_hash"]
        if raw["output_hash"] != output_hash:
            raise RuntimeError("Routing diagnostic output hashes differ")
        routing.append({"phase": phase, "program_counts": counts})
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".json", ".log", ".guard.json"))
        evidence.append(profile)
        print(json.dumps(routing[-1]), flush=True)
    if routing[0]["program_counts"] != routing[1]["program_counts"]:
        raise RuntimeError("Control/candidate batch dispatch counts differ")
    times, peaks = [], []
    for flag, candidate, phase in ((0, False, "before"), (1, True, "candidate"), (0, False, "after")):
        prefix = ROOT / f"{VARIANT}-operator-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(prefix, flag, candidate))
        child.check_returncode()
        raw = json.loads(prefix.with_suffix(".json").read_text())
        prior.validate_operator(raw, guard, False)
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
        "status": "qmoe_early_scratch_batch_operator_screen_completed",
        "provider_sha256": build["provider_sha256"],
        "baseline_provider_sha256": build["baseline_provider_sha256"],
        "native_qualification_reused_without_retry": str(native_path),
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
