import json
import math
from pathlib import Path
import re

from build_qmoe_node_replay import sha
from gpu_timestamp_ns_transform import precise_duration_us
from prepare_hc_up_zero_tail_actual_v3 import range_sha
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_dawn_scope_qualification_v1 import PARENT, REPOS, command_for as base_command, container
from run_fc2_rank_waves_native import guarded


BUILD = REPOS / "qwen38-int4-gpu-timestamp-ns/build-v1"
BINARY = REPOS / "qwen38-int4-hc-up-packed160/build-v1/hc_up_compare"
REFERENCE = REPOS / "qwen38-int4-hc-up-packed160/actual-screen-v1"
FIXTURE = REPOS / "qwen38-int4-hc-up-zero-tail/actual-fixtures-v3/anchor0-rows1.onnx"
VARIANT = "gpu-timestamp-ns-v1"


def validate_profile(events, precise):
    kernels = [event for event in events if event.get("cat") == "Kernel"]
    if len(kernels) != 288 or any(
            not event.get("name", "").endswith("&HCUpPreMixPacked2F16")
            or event.get("pid") != -1 or event.get("tid") != -1 or event.get("ph") != "X"
            or not math.isfinite(event.get("ts", -1)) or event["ts"] < 0
            or not math.isfinite(event.get("dur", -1)) or event["dur"] < 0
            or ":128,1,1:" not in event.get("args", {}).get("cache_key", "")
            or not event.get("args", {}).get("shapes")
            for event in kernels):
        raise RuntimeError("Wrong precise-timestamp original compute path or native scope")
    if not precise:
        if any("gpu_duration_ns" in event["args"] for event in kernels):
            raise RuntimeError("Original provider unexpectedly has private nanosecond metadata")
        return {"kernel_events": len(kernels), "nanosecond_metadata": False}
    values = [precise_duration_us(event) for event in kernels]
    fractional = sum(int(event["args"]["gpu_duration_ns"]) % 1000 != 0 for event in kernels)
    if fractional == 0:
        raise RuntimeError("Resolved GPU timestamps do not provide sub-microsecond discrimination")
    return {"kernel_events": len(kernels), "nanosecond_metadata": True,
            "fractional_microsecond_events": fractional, "minimum_gpu_us": min(values),
            "maximum_gpu_us": max(values)}


def command_for(provider, prefix):
    arguments = ["compare", container(REFERENCE), container(Path(str(prefix) + "-profile")), container(FIXTURE)]
    command = base_command(BINARY, provider, arguments)
    command[-1] = ("export ORT_WEBGPU_HC_UP_PACKED160=0 ORT_WEBGPU_HC_UP_PREMIX=1 "
                   "ORT_WEBGPU_HC_UP_PARALLEL2=1 ORT_TEST_HC_UP_NATIVE_PROFILE=1 "
                   "ORT_TEST_HC_UP_FP32_ACCUMULATION=0; " + command[-1])
    return command


def main():
    output = ROOT / f"{VARIANT}-qualification.json"
    if output.exists():
        raise FileExistsError("Fresh timestamp precision qualification required")
    manifest_path = BUILD / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (not manifest["original_context_object_byte_identical"] or not manifest["baseline_byte_identical"]
            or not manifest["compute_code_and_all_other_objects_unchanged"]
            or manifest["baseline_provider_sha256"] !=
            "3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3"):
        raise RuntimeError("Missing exact original-context/provider and additive-only source proof")
    for path, digest in manifest["source_sha256"].items():
        resolved = host_path(path) if path.startswith(("/workspace/", "/evidence/")) else Path(path)
        if sha(resolved) != digest:
            raise RuntimeError(f"Changed timestamp build input: {path}")
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    original = PARENT / "libonnxruntime_providers_webgpu.so"
    if sha(provider) != manifest["provider_sha256"] or sha(original) != manifest["baseline_provider_sha256"]:
        raise RuntimeError("Changed timestamp or original provider")
    previous_path = ROOT / "hc-up-packed160-v1-outcome.json"
    previous = json.loads(previous_path.read_text())
    generation_path = ROOT / "hc-up-zero-tail-actual-v3-generation.json"
    if sha(generation_path) != previous["source_and_evidence_sha256"][str(generation_path)]:
        raise RuntimeError("Changed original tensor provenance")
    generation = json.loads(generation_path.read_text())
    if sha(Path(generation["model"])) != generation["model_sha256"]:
        raise RuntimeError("Changed frozen model graph")
    for record in generation["original_tensor_ranges"]:
        source, alias = Path(record["source"]), Path(record["alias"])
        if (not alias.samefile(source)
                or range_sha(source, record["offset"], record["length"]) != record["range_sha256"]):
            raise RuntimeError("Changed frozen tensor bytes or original-inode alias")
    reference = REFERENCE / (FIXTURE.name + ".bin")
    for path in (BINARY, FIXTURE, reference):
        if sha(path) != previous["source_and_evidence_sha256"][str(path)]:
            raise RuntimeError(f"Changed independent original fixture/comparator/golden: {path}")
    evidence = [Path(__file__), ROOT / "gpu_timestamp_ns_transform.py", GUARD, manifest_path,
                previous_path, generation_path, BINARY, FIXTURE, reference, provider, original]
    records, fingerprint = [], None
    for phase, library, precise in (("original", original, False), ("precise-original-compute", provider, True)):
        prefix = ROOT / f"{VARIANT}-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(library, prefix))
        child.check_returncode()
        text = prefix.with_suffix(".log").read_text()
        rows = re.findall(
            r"^PASS (\S+) comparisons=(\d+) fingerprint=(\d+) profiling=1 fp32=0 pending=64 "
            r"warmup_pairs=16 measured_pairs=128 major_faults=0$", text, re.M)
        profiles = re.findall(r"^PROFILE (\S+)$", text, re.M)
        if (guard["status"] != "completed" or "FAIL" in text or len(rows) != 1
                or rows[0][:2] != (FIXTURE.name, str(2560 * 144 * 2)) or len(profiles) != 1):
            raise RuntimeError("Incomplete independent original-golden precision proof")
        if fingerprint is None:
            fingerprint = rows[0][2]
        elif fingerprint != rows[0][2]:
            raise RuntimeError("Additive timestamp metadata changed original output fingerprints")
        profile = host_path(profiles[0])
        metadata = validate_profile(json.loads(profile.read_text()), precise)
        records.append({"phase": phase, "finite_bitwise_comparisons": 2560 * 144 * 2,
                        "sampled_peak_gpu_gib": guard["peak_gpu_used_bytes"] / 2**30, **metadata})
        evidence.extend([profile, *[prefix.with_suffix(suffix) for suffix in (".log", ".guard.json")]])
    report = {"status": "additive_gpu_nanosecond_metadata_qualified", "precision_qualified": True,
              "phases": records, "provider_sha256": manifest["provider_sha256"],
              "compute_definition": manifest["control_definition"],
              "no_held_candidate_retest": True, "no_global_settings_changes": True,
              "performance_qualified": False, "quality_qualified": False,
              "source_and_evidence_sha256": {str(path): sha(path) for path in evidence}}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "source_and_evidence_sha256"}), flush=True)


if __name__ == "__main__":
    main()
