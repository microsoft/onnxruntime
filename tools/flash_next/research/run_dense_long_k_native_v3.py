import json
import math
from pathlib import Path
import re

from build_qmoe_node_replay import sha
from dense_tile_parallel3_long_k_transform import FLAG, NAME
from dense_tile_parallel3_transform import PROGRAM
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_dawn_scope_qualification_v1 import PARENT, REPOS, command_for as base_command, container
from run_fc2_rank_waves_native import guarded
from run_dense_tile_parallel3_v1 import matched


BUILD = REPOS / "qwen38-int4-dense-tile-parallel3-long-k/build-v1"
NATIVE = BUILD.parent / "native-full-v3"
DATA = BUILD.parent / "native-proof-v3"
FIXTURES = REPOS / "qwen38-int4-generic-matmul-replay/fixtures-v1"
VARIANT = "dense-tile-parallel3-long-k-native-v3"
TOTAL = 116582784


def shape(path):
    match = re.fullmatch(r"matmul_k(\d+)_n(\d+)_rows(\d+)\.onnx", Path(path).name)
    if match is None:
        raise ValueError("Unknown frozen dense fixture name")
    return tuple(map(int, match.groups()))


def verify_profile(events, selected, calls):
    kernels = [event for event in events if event.get("cat") == "Kernel"]
    if len(kernels) < calls:
        raise RuntimeError("Missing native GPU fallback or selected events")
    private = [event for event in kernels if event.get("name", "").endswith("&" + NAME)]
    if len(private) != (calls if selected else 0):
        raise RuntimeError("Wrong long-K-only actual GPU coverage")
    for event in private:
        if (event.get("pid") != -1 or event.get("tid") != -1 or event.get("ph") != "X"
                or not math.isfinite(event.get("dur", 0)) or event.get("dur", 0) <= 0
                or ":384,1,1:" not in event.get("args", {}).get("cache_key", "")):
            raise RuntimeError("Wrong selected native GPU event/workgroup")


def verify_replay(rows, fp32, requests, profiling):
    if fp32 or profiling:
        if rows:
            raise RuntimeError("FP32 accumulation unexpectedly entered private/ordinary replay route")
        return
    if len(rows) != 9 or any(
            row["operator"] != "MatMulNBitsExact" or row["requests"] != requests
            or row["hits"] < requests - 16 or row["binding_hits"] < requests - 16
            or row["captures"] < 1 for row in rows):
        raise RuntimeError("Incomplete real dense replay/binding reuse")


def command_for(provider, mode, directory, prefix, fixtures, selected, fp32, profiling=1):
    if (selected not in (0, 1, 2) or fp32 not in (0, 1) or profiling not in (0, 1)
            or mode not in ("capture", "compare")):
        raise ValueError("Invalid fixed-process native launch")
    stats = Path(str(prefix) + "-stats.jsonl")
    command = base_command(
        NATIVE / "native_compare", provider,
        [mode, container(directory), container(prefix), *[container(path) for path in fixtures]], stats)
    command[-1] = (
        f"export {FLAG}={selected} ORT_TEST_DENSE_FP32_ACCUMULATION={fp32} "
        f"ORT_TEST_DENSE_NATIVE_PROFILE={profiling} "
        "ORT_WEBGPU_DENSE_COLUMNS8_DOUBLE_BUFFER=1 ORT_WEBGPU_DENSE_BLOCK32=1 "
        "ORT_WEBGPU_DENSE_TILE4=1 ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1; " + command[-1])
    return command


def main():
    output = ROOT / f"{VARIANT}-outcome.json"
    if output.exists() or DATA.exists():
        raise FileExistsError("Fresh full native prefix required")
    manifest_path = BUILD / "build-manifest.json"
    generation_path = NATIVE / "generation.json"
    manifest = json.loads(manifest_path.read_text())
    generation = json.loads(generation_path.read_text())
    if (not manifest["baseline_byte_identical"] or not manifest["original_dense_object_byte_identical"]
            or manifest["supported_k"] != [6144] or manifest["supported_n"] != 2560):
        raise RuntimeError("Missing supported long-K-only build proof")
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    original = PARENT / "libonnxruntime_providers_webgpu.so"
    if (sha(provider) != manifest["provider_sha256"]
            or sha(original) != manifest["baseline_provider_sha256"]
            or manifest["baseline_provider_sha256"] !=
            "3a283ecfb892613b19a6f49d15b726e928b41027c8b3d838bd8cd1dfed9939d3"
            or sha(NATIVE / "native_compare") != generation["binary_sha256"]):
        raise RuntimeError("Changed supported provider or fixed-process comparator")
    for records in (manifest["source_sha256"], generation["source_sha256"]):
        for path, digest in records.items():
            resolved = host_path(path) if path.startswith(("/workspace/", "/evidence/")) else Path(path)
            if sha(resolved) != digest:
                raise RuntimeError(f"Changed qualification input: {path}")
    if PROGRAM not in (BUILD / "matmul_nbits.cc").read_text():
        raise RuntimeError("GPU program differs from the independently measured kernel")
    evidence = [
        Path(__file__), GUARD, ROOT / "run_dense_tile_parallel3_v1.py",
        ROOT / "run_dawn_scope_qualification_v1.py", ROOT / "run_fc2_rank_waves_native.py",
        ROOT / "dense_tile_parallel3_transform.py", ROOT / "dense_tile_parallel3_narrow_transform.py",
        ROOT / "dense_tile_parallel3_long_k_transform.py", manifest_path, generation_path,
        provider, original, NATIVE / "native_compare",
    ]
    operator_proofs = []
    for filename in ("dense-tile-parallel3-v1-outcome.json",
                     "dense-tile-parallel3-narrow-v1-outcome.json"):
        path = ROOT / filename
        proof = json.loads(path.read_text())
        windows = {}
        for row in proof["records"]:
            if row.get("k") == 6144 and row.get("n") == 2560:
                windows[row["phase"]] = row["gpu_us"]
        result = matched(windows["before"], windows["selected"], windows["after"])
        if not result["operator_gate_pass"]:
            raise RuntimeError("Long-K shader lacks consistent independent operator support")
        operator_proofs.append({"evidence": str(path), **result})
        evidence.append(path)
    frozen_path = ROOT / "dense-double-buffer-v1-fp32-native-gpu7-20261030.qualification.json"
    frozen = json.loads(frozen_path.read_text())
    fixtures = sorted(FIXTURES.glob("matmul_*.onnx"))
    if len(fixtures) != 27:
        raise RuntimeError("Expected exactly 27 original dense fixtures")
    for fixture in fixtures:
        if sha(fixture) != frozen["fixture_sha256"][fixture.name]:
            raise RuntimeError(f"Changed original fixture: {fixture}")
    if sum(shape(path)[1] * shape(path)[2] * 128 for path in fixtures) != TOTAL:
        raise RuntimeError("Wrong exact native output coverage")
    evidence.extend([frozen_path, *fixtures])
    DATA.mkdir()
    for directory in ("fp16", "fp32", "invalid"):
        (DATA / directory).mkdir()
    prefix = ROOT / f"{VARIANT}-invalid-gpu0"
    child, guard = guarded(prefix, command_for(
        provider, "capture", DATA / "invalid", prefix, fixtures[:1], 2, 0))
    text = prefix.with_suffix(".log").read_text()
    if child.returncode != 1 or f"{FLAG} must be 0 or 1" not in text:
        raise RuntimeError("Invalid long-K private selector did not reject explicitly")
    evidence.extend(prefix.with_suffix(suffix) for suffix in (".log", ".guard.json"))
    phases, originals, fingerprints = [], {}, {}
    for phase, library, mode, selected, fp32, profiling in (
            ("original", original, "capture", 0, 0, 1), ("default-off", provider, "compare", 0, 0, 1),
            ("selected", provider, "compare", 1, 0, 1), ("original-after", original, "compare", 0, 0, 1),
            ("fp32-original", original, "capture", 0, 1, 1), ("fp32-selected", provider, "compare", 1, 1, 1),
            ("replay-original", original, "compare", 0, 0, 0),
            ("replay-default-off", provider, "compare", 0, 0, 0),
            ("replay-selected", provider, "compare", 1, 0, 0),
            ("replay-original-after", original, "compare", 0, 0, 0),
            ("replay-fp32-original", original, "compare", 0, 1, 0),
            ("replay-fp32-selected", provider, "compare", 1, 1, 0)):
        directory = DATA / ("fp32" if fp32 else "fp16")
        if mode == "compare":
            for reference, digest in originals[fp32].items():
                if sha(Path(reference)) != digest:
                    raise RuntimeError("Independent original full-native golden changed")
        prefix = ROOT / f"{VARIANT}-{phase}-gpu0"
        child, guard = guarded(prefix, command_for(library, mode, directory, prefix, fixtures, selected, fp32, profiling))
        child.check_returncode()
        text = prefix.with_suffix(".log").read_text()
        rows = re.findall(
            r"^PASS (\S+) changing_rounds=128 finite_bitwise_elements=(\d+) fingerprint=(\d+) pending=(1|64) fp32=(0|1) profiling=(0|1)$",
            text, re.M)
        profiles = re.findall(r"^PROFILE (\S+)$", text, re.M)
        if (guard["status"] != "completed" or "FAIL" in text or len(rows) != 27 or len(profiles) != (27 if profiling else 0)
                or f"PASS total_exact_elements={TOTAL}; CPU fallback disabled; fixed_process=1" not in text):
            raise RuntimeError("Incomplete independent full-native finite/golden proof")
        for index, (row, profile_name, fixture) in enumerate(zip(rows, profiles if profiling else [None] * 27, fixtures, strict=True)):
            k, n, count_rows = shape(fixture)
            if (row[0] != fixture.name or int(row[1]) != n * count_rows * 128
                    or row[3] != ("1" if index % 2 == 0 else "64") or int(row[4]) != fp32
                    or int(row[5]) != profiling):
                raise RuntimeError("Wrong fixture rank/count or FP32 fallback scope")
            if profile_name is not None:
                profile = host_path(profile_name)
                verify_profile(json.loads(profile.read_text()),
                               bool(selected and not fp32 and (k, n, count_rows) == (6144, 2560, 1)),
                               256 if mode == "capture" else 128)
                evidence.append(profile)
            reference = directory / (fixture.name + ".bin")
            if reference.stat().st_size != n * count_rows * 128 * 2:
                raise RuntimeError("Wrong native golden byte count")
            evidence.append(reference)
        stats = Path(str(prefix) + "-stats.jsonl")
        if (fp32 or profiling) and not stats.exists():
            replay = []
        else:
            replay = [json.loads(line) for line in stats.read_text().splitlines()]
            evidence.append(stats)
        verify_replay(replay, fp32, 256 if mode == "capture" else 128, profiling)
        if mode == "capture":
            originals[fp32] = {str(directory / (fixture.name + ".bin")):
                              sha(directory / (fixture.name + ".bin")) for fixture in fixtures}
            fingerprints[fp32] = rows
        elif [(row[0], row[1], row[2], row[4]) for row in rows] != [
                (row[0], row[1], row[2], row[4]) for row in fingerprints[fp32]]:
            raise RuntimeError("Independent full-native output fingerprints differ")
        phases.append({
            "phase": phase, "fixtures": 27, "finite_bitwise_comparisons": TOTAL,
            "fp32_accumulation_fallback": bool(fp32), "replay_instances": len(replay),
            "profiling_enabled": bool(profiling),
            "expected_single_row_replay_instances": 9 if not fp32 and not profiling else 0,
            "sampled_peak_gpu_gib": guard["peak_gpu_used_bytes"] / 2**30,
            "all_independent_original_output_fingerprints_agree": True,
        })
        evidence.extend(prefix.with_suffix(suffix) for suffix in (".log", ".guard.json"))
        print(json.dumps(phases[-1]), flush=True)
    report = {
        "status": "dense_long_k_fixed_process_native_golden_qualified",
        "provider_sha256": manifest["provider_sha256"], "phases": phases,
        "invalid_private_selector_rejected": True, "operator_support_evidence": operator_proofs,
        "native_model_admission_eligible": True, "performance_qualified": False,
        "quality_qualified": False, "default_enabled": False,
        "source_and_evidence_sha256": {str(path): sha(path) for path in set(evidence)},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "source_and_evidence_sha256"}), flush=True)


if __name__ == "__main__":
    main()
