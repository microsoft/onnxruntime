import json
import math
from pathlib import Path
import re
import statistics

from build_qmoe_node_replay import sha
from hc_up_zero_tail_transform import FLAG, NAME
from prepare_hc_up_zero_tail_actual_v2 import range_sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_dawn_scope_qualification_v1 import PARENT, REPOS, command_for as base_command, container
from run_fc2_rank_waves_native import guarded


BUILD = REPOS / "qwen38-int4-hc-up-zero-tail/build-v1"
BINARY = BUILD / "hc_up_compare"
DATA = BUILD.parent / "actual-screen-v2"
VARIANT = "hc-up-zero-tail-v2"


def command_for(provider, mode, prefix, selected, fixture):
    if mode not in ("capture", "compare") or selected not in (0, 1, 2):
        raise ValueError("Invalid native screen mode or selector")
    reference_directory = DATA / "invalid" if selected == 2 else DATA
    arguments = [mode, container(reference_directory), container(Path(str(prefix) + "-profile")), container(fixture)]
    command = base_command(BINARY, provider, arguments)
    command[-1] = (f"export {FLAG}={selected} ORT_WEBGPU_HC_UP_PREMIX=1 "
                   "ORT_WEBGPU_HC_UP_PARALLEL2=1 ORT_TEST_HC_UP_NATIVE_PROFILE=1 "
                   "ORT_TEST_HC_UP_FP32_ACCUMULATION=0; " + command[-1])
    return command


def measured_gpu(events, selected):
    name = NAME if selected else "HCUpPreMixPacked2F16"
    kernels = [event for event in events if event.get("cat") == "Kernel"]
    if len(kernels) != 288 or any(
            not event.get("name", "").endswith("&" + name)
            or event.get("pid") != -1 or event.get("tid") != -1 or event.get("ph") != "X"
            or not math.isfinite(event.get("dur", 0)) or event["dur"] <= 0
            or ":128,1,1:" not in event.get("args", {}).get("cache_key", "")
            for event in kernels):
        raise RuntimeError("Wrong HC-up GPU program, timestamp, paired scope or workgroup")
    return [event["dur"] for event in kernels[33::2]]


def matched(before, candidate, after):
    if any(len(values) != 128 for values in (before, candidate, after)):
        raise ValueError("Expected 128 matched GPU calls after 16 excluded warmup pairs")
    if any(not math.isfinite(value) or value <= 0 for values in (before, candidate, after) for value in values):
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
        raise FileExistsError("Fresh HC-up screen prefix required")
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (not manifest["baseline_byte_identical"] or not manifest["original_hc_up_object_byte_identical"]
            or manifest["physical_threads"] != 128 or manifest["shared_storage_bytes"] != 528
            or manifest["flag"] != FLAG or manifest["program"] != NAME):
        raise RuntimeError("Missing supported original provider/object proof or geometry")
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
    generation_path = ROOT / "hc-up-zero-tail-actual-v2-generation.json"
    generation = json.loads(generation_path.read_text())
    if sha(Path(generation["model"])) != generation["model_sha256"]:
        raise RuntimeError("Changed frozen model graph")
    for record in generation["original_tensor_ranges"]:
        source, alias = Path(record["source"]), Path(record["alias"])
        if (not alias.samefile(source)
                or range_sha(source, record["offset"], record["length"]) != record["range_sha256"]):
            raise RuntimeError("Changed original model tensor range or alias")
    fixtures = [record for record in generation["fixtures"] if record["anchor"] == 0 and record["rows"] == 1]
    if len(fixtures) != 1:
        raise RuntimeError("Missing unique first actual-weight single-row fixture")
    fixture = Path(fixtures[0]["path"])
    if sha(fixture) != fixtures[0]["sha256"]:
        raise RuntimeError("Changed actual-weight fixture")
    DATA.mkdir()
    (DATA / "invalid").mkdir()
    reference = DATA / (fixture.name + ".bin")
    evidence = [Path(__file__), ROOT / "hc_up_zero_tail_transform.py", GUARD, generation_path,
                manifest_path, BINARY, provider, original, fixture]
    records, windows, fingerprint = [], {}, None

    def save_execution(status):
        execution.write_text(json.dumps({"status": status, "records": records}, indent=2) + "\n")

    save_execution("running")
    prefix = ROOT / f"{VARIANT}-invalid-gpu0"
    child, guard = guarded(prefix, command_for(provider, "capture", prefix, 2, fixture))
    text = prefix.with_suffix(".log").read_text()
    invalid_reference = DATA / "invalid" / reference.name
    if (child.returncode != 1 or f"{FLAG} must be 0 or 1" not in text
            or (invalid_reference.exists() and invalid_reference.stat().st_size != 0)
            or reference.exists()):
        raise RuntimeError("Invalid selector did not reject before producing golden values")
    records.append({"phase": "invalid", "invalid_selector_rejected": True})
    evidence.extend(prefix.with_suffix(suffix) for suffix in (".log", ".guard.json"))
    save_execution("running")
    reference_sha = None
    for phase, library, mode, selected in (
            ("before", original, "capture", 0), ("default-off", provider, "compare", 0),
            ("selected", provider, "compare", 1), ("after", original, "compare", 0)):
        if mode == "compare" and sha(reference) != reference_sha:
            raise RuntimeError("Independent original golden bytes changed")
        prefix = ROOT / f"{VARIANT}-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(library, mode, prefix, selected, fixture))
        child.check_returncode()
        text = prefix.with_suffix(".log").read_text()
        rows = re.findall(
            r"^PASS (\S+) comparisons=(\d+) fingerprint=(\d+) profiling=1 fp32=0 pending=64 "
            r"warmup_pairs=16 measured_pairs=128 major_faults=0$", text, re.M)
        profiles = re.findall(r"^PROFILE (\S+)$", text, re.M)
        if (guard["status"] != "completed" or "FAIL" in text or len(rows) != 1
                or rows[0][:2] != (fixture.name, str(2560 * 144 * 2))
                or len(profiles) != 1 or reference.stat().st_size != 2560 * 144 * 2):
            raise RuntimeError("Incomplete actual-weight finite native proof or measured scope")
        if fingerprint is None:
            fingerprint = rows[0][2]
            reference_sha = sha(reference)
            evidence.append(reference)
        elif fingerprint != rows[0][2]:
            raise RuntimeError("Independent original/default/selected/after fingerprints differ")
        profile = host_path(profiles[0])
        durations = measured_gpu(json.loads(profile.read_text()), bool(selected))
        windows[phase] = durations
        records.append({"phase": phase, "finite_bitwise_comparisons": 2560 * 144 * 2,
                        "median_gpu_us": statistics.median(durations), "gpu_us": durations,
                        "measured_major_faults": 0,
                        "sampled_peak_gpu_gib": guard["peak_gpu_used_bytes"] / 2**30})
        evidence.extend([profile, *[prefix.with_suffix(suffix) for suffix in (".log", ".guard.json")]])
        save_execution("running")
    comparison = matched(windows["before"], windows["selected"], windows["after"])
    save_execution("completed")
    report = {
        "status": "hc_up_zero_tail_actual_operator_screen_completed",
        "operator": comparison, "phases": records,
        "held": not comparison["operator_gate_pass"], "no_unchanged_retry": True,
        "full_native_admission_eligible": comparison["operator_gate_pass"],
        "model_admission_eligible": False, "performance_qualified": False, "quality_qualified": False,
        "peak_gpu_gib": max(record.get("sampled_peak_gpu_gib", 0) for record in records),
        "provider_sha256": manifest["provider_sha256"], "flag": FLAG, "program": NAME,
        "source_and_evidence_sha256": {str(path): sha(path) for path in evidence},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "source_and_evidence_sha256" and key != "phases"}), flush=True)


if __name__ == "__main__":
    main()
