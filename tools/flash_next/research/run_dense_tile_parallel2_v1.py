import json
import math
from pathlib import Path
import re
import shlex
import statistics

from build_qmoe_node_replay import replace_once, sha
from dense_tile_parallel2_transform import FLAG, NAME
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_dawn_scope_qualification_v1 import PARENT, REPOS, command_for as base_command, container
from run_fc2_rank_waves_native import guarded


BUILD = REPOS / "qwen38-int4-dense-tile-parallel2/build-v1"
BINARY = BUILD / "dense_compare"
DATA = BUILD.parent / "actual-screen-v1"
FIXTURES = REPOS / "qwen38-int4-dense-paired-matrix-reset/actual-fixtures-v1"
CASES = ((640, 2560), (2560, 2560), (2560, 7168), (6144, 2560),
         (2560, 16384), (2560, 6144), (2560, 10240))
VARIANT = "dense-tile-parallel2-v1"


def command_for(mode, provider, fixture, reference, prefix, selected):
    if mode not in ("capture", "compare") or selected not in (0, 1, 2):
        raise ValueError("Invalid dense screen mode or selector")
    arguments = [mode, container(fixture), container(reference), container(Path(str(prefix) + "-profile"))]
    command = base_command(BINARY, provider, arguments)
    original = "exec " + shlex.join([container(BINARY), container(provider), *arguments])
    replacement = "exec " + shlex.join([container(BINARY), mode, container(provider), *arguments[1:]])
    command[-1] = replace_once(command[-1], original, replacement)
    command[-1] = (
        f"export {FLAG}={selected} ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER=1 "
        "ORT_WEBGPU_DENSE_BLOCK32=1 ORT_WEBGPU_DENSE_TILE4=1 "
        "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1; " + command[-1])
    return command


def measured_gpu(events, selected):
    name = NAME if selected else "DenseColumns8DoubleBuffer"
    threads = 256 if selected else 128
    kernels = [event for event in events if event.get("cat") == "Kernel"]
    if len(kernels) != 144 or any(
            not event.get("name", "").endswith("&" + name)
            or event.get("pid") != -1 or event.get("tid") != -1 or event.get("ph") != "X"
            or not math.isfinite(event.get("dur", 0)) or event["dur"] <= 0
            or f":{threads},1,1:" not in event.get("args", {}).get("cache_key", "")
            for event in kernels):
        raise RuntimeError("Wrong dense actual GPU program, scope, timestamp or workgroup")
    return [event["dur"] for event in kernels[16:]]


def matched(before, candidate, after):
    if any(len(values) != 128 for values in (before, candidate, after)):
        raise ValueError("Expected 128 matched actual GPU calls")
    if any(not math.isfinite(value) or value <= 0 for row in (before, candidate, after) for value in row):
        raise ValueError("Invalid actual GPU duration")
    ratios = [math.sqrt(a * b) / c for a, c, b in zip(before, candidate, after, strict=True)]
    ratio = statistics.geometric_mean(ratios)
    positive = sum(value > 1 for value in ratios)
    return {"matched_gpu_speed_ratio": ratio, "positive_gpu_calls": positive,
            "operator_gate_pass": ratio > 1.01 and positive >= 120}


def main():
    output = ROOT / f"{VARIANT}-outcome.json"
    execution = ROOT / f"{VARIANT}-execution.json"
    if output.exists() or execution.exists() or DATA.exists():
        raise FileExistsError("Fresh dense screen prefix required")
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (not manifest["baseline_byte_identical"] or not manifest["original_dense_object_byte_identical"]
            or manifest["physical_threads"] != 256 or manifest["shared_storage_bytes"] != 11264):
        raise RuntimeError("Missing supported original dense/provider proof or geometry")
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    original = PARENT / "libonnxruntime_providers_webgpu.so"
    if (sha(provider) != manifest["provider_sha256"]
            or sha(original) != manifest["baseline_provider_sha256"]
            or manifest["baseline_provider_sha256"] !=
            "3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3"):
        raise RuntimeError("Changed original or candidate provider")
    for path, digest in manifest["source_sha256"].items():
        resolved = host_path(path) if path.startswith(("/workspace/", "/evidence/")) else Path(path)
        if sha(resolved) != digest:
            raise RuntimeError(f"Changed build input: {path}")
    frozen_path = ROOT / "dense-paired-matrix-reset-actual-v1-outcome.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen["status"] != "packed_dense_original_provider_actual_weights_supported":
        raise RuntimeError("Actual matrix fixtures lack original-provider provenance")
    fixtures = {case: FIXTURES / f"k{case[0]}-n{case[1]}.onnx" for case in CASES}
    for fixture in fixtures.values():
        if sha(fixture) != frozen["source_and_evidence_sha256"][str(fixture)]:
            raise RuntimeError(f"Changed actual-weight fixture: {fixture}")
    DATA.mkdir()
    evidence = [Path(__file__), ROOT / "dense_tile_parallel2_transform.py", GUARD,
                manifest_path, provider, original, BINARY, frozen_path, *fixtures.values()]
    records, captures, windows = [], {}, {}

    def save_execution():
        execution.write_text(json.dumps({"status": "running", "records": records}, indent=2) + "\n")

    save_execution()
    prefix = ROOT / f"{VARIANT}-invalid-gpu0"
    child, guard = guarded(prefix, command_for(
        "capture", provider, fixtures[CASES[0]], DATA / "invalid-unused.f16", prefix, 2))
    text = prefix.with_suffix(".log").read_text()
    if (child.returncode != 1 or f"{FLAG} must be 0 or 1" not in text
            or (DATA / "invalid-unused.f16").exists()):
        raise RuntimeError("Invalid dense selector did not reject explicitly")
    records.append({"phase": "invalid", "invalid_selector_rejected": True})
    evidence.extend(prefix.with_suffix(suffix) for suffix in (".log", ".guard.json"))
    save_execution()
    for phase, library, mode, selected in (
            ("before", original, "capture", 0), ("default-off", provider, "compare", 0),
            ("selected", provider, "compare", 1), ("after", original, "compare", 0)):
        for case, fixture in fixtures.items():
            k, n = case
            reference = DATA / f"k{k}-n{n}-reference.f16"
            if mode == "capture":
                if reference.exists():
                    raise FileExistsError(reference)
            elif sha(reference) != captures[case]:
                raise RuntimeError("Independent finite reference changed")
            prefix = ROOT / f"{VARIANT}-{phase}-k{k}-n{n}-gpu0"
            child, guard = guarded(prefix, command_for(mode, library, fixture, reference, prefix, selected))
            child.check_returncode()
            text = prefix.with_suffix(".log").read_text()
            row = re.findall(r"^DENSE comparisons=(\d+) mismatches=(\d+) K=(\d+) N=(\d+)$", text, re.M)
            profiles = re.findall(r"^PROFILE (\S+)$", text, re.M)
            if (guard["status"] != "completed" or "FAIL" in text
                    or row != [(str(n * 144), "0", str(k), str(n))]
                    or len(profiles) != 1 or "SCOPE warmups=16 measured=128 major_faults=0" not in text
                    or reference.stat().st_size != n * 36 * 2):
                raise RuntimeError("Incomplete finite original/default/selected dense scope")
            if mode == "capture":
                captures[case] = sha(reference)
                evidence.append(reference)
            profile = host_path(profiles[0])
            durations = measured_gpu(json.loads(profile.read_text()), bool(selected and k in (2560, 6144)))
            windows[(phase, case)] = durations
            records.append({
                "phase": phase, "k": k, "n": n, "finite_bitwise_comparisons": n * 144,
                "median_gpu_us": statistics.median(durations), "gpu_us": durations,
                "measured_major_faults": 0, "sampled_peak_gpu_gib": guard["peak_gpu_used_bytes"] / 2**30,
            })
            evidence.extend([profile, *[prefix.with_suffix(suffix) for suffix in (".log", ".guard.json")]])
            save_execution()
            print(json.dumps({key: value for key, value in records[-1].items() if key != "gpu_us"}), flush=True)
    shapes = [
        {"k": k, "n": n, "selected": k in (2560, 6144),
         **matched(windows[("before", (k, n))], windows[("selected", (k, n))],
                   windows[("after", (k, n))])}
        for k, n in CASES]
    supported = all(row["operator_gate_pass"] for row in shapes if row["selected"])
    report = {
        "status": "dense_tile_parallel3_actual_operator_screen_completed",
        "provider_sha256": manifest["provider_sha256"], "shapes": shapes, "records": records,
        "operator_support": supported, "held_no_unchanged_retry": not supported,
        "model_admission_eligible": False, "full_native_fallback_qualification_required": True,
        "performance_qualified": False, "quality_qualified": False, "default_enabled": False,
        "operator_ratios_are_not_model_decode_ratios": True,
        "source_and_evidence_sha256": {str(path): sha(path) for path in evidence},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    execution.write_text(json.dumps({"status": "completed", "records": records}, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("records", "source_and_evidence_sha256")}), flush=True)


if __name__ == "__main__":
    main()
