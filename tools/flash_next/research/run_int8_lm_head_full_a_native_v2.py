import collections
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from build_int8_lm_head_full_a import FLAG, NAME
from build_qmoe_node_replay import sha
from prepare_int8_lm_head_full_a_native_v2 import DATA, MODEL, PROOF, region_sha
from run_dense_columns16_synthetic import ROOT, run_guard, UUID, workspace
from run_hc_narrow_rms_broadcast_native import host_path


BUILD = Path("/home/sayanshaw/repos/qwen38-int4-int8-lm-head-full-a/build-v1")
ORIGINAL = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/libonnxruntime_providers_webgpu.so")
REFERENCES = PROOF / "original-references-v2"
FLAGS = (
    "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
    "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
    "ORT_WEBGPU_GENERIC_MATMUL_REPLAY=1 ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 "
    "ORT_WEBGPU_DENSE_BLOCK32=1"
)


if __name__ == "__main__":
    outcome = ROOT / "int8-lm-head-full-a-v2-native-outcome.json"
    execution = ROOT / "int8-lm-head-full-a-v2-native-execution.json"
    if outcome.exists() or execution.exists() or REFERENCES.exists():
        raise FileExistsError("Existing independent INT8 head evidence")
    build_path, comparator_path = BUILD / "build-manifest.json", PROOF / "build-manifest.json"
    build, comparator = (json.loads(path.read_text()) for path in (build_path, comparator_path))
    assert build["baseline_byte_identical"] and build["all_152_original_source_objects_byte_identical"]
    assert comparator["status"] == "typed_int8_head_comparator_built_io_tested"
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == build["provider_sha256"] and sha(ORIGINAL) == build["baseline_provider_sha256"]
    evidence = [Path(__file__), build_path, comparator_path, provider, ORIGINAL]
    for name, digest in build["source_sha256"].items():
        path = BUILD / name
        assert sha(path) == digest, path
        evidence.append(path)
    for name, digest in comparator["source_sha256"].items():
        path = host_path(name)
        assert sha(path) == digest, path
        evidence.append(path)
    fixture_path = DATA / "manifest.json"
    manifest = json.loads(fixture_path.read_text())
    assert sha(MODEL / "model.onnx") == manifest["graph_sha256"]
    assert sha(MODEL / "genai_config.json") == manifest["config_sha256"]
    assert (DATA / "model.onnx.data").samefile(MODEL / "model.onnx.data")
    fixtures = manifest["fixtures"]
    assert len(fixtures) == 13 and sum(row["actual_weights"] for row in fixtures) == 1
    for row in fixtures:
        assert sha(Path(row["path"])) == row["sha256"]
        for tensor in row.get("tensors", []):
            external = tensor["external"]
            assert region_sha(MODEL / external["location"], int(external["offset"]), int(external["length"])) == tensor["payload_sha256"]
    evidence.extend([fixture_path, MODEL / "model.onnx", MODEL / "genai_config.json",
                     *[Path(row["path"]) for row in fixtures]])
    if shutil.disk_usage(PROOF).free < 256 * 1024**2:
        raise RuntimeError("Insufficient bounded original-reference space")
    REFERENCES.mkdir()
    records, captured, coverage = [], {}, []
    expected = sum(row["m"] * row["n"] * 64 for row in fixtures)
    status = "int8_head_separate_original_outputs_supported"
    for phase, library, mode, selected in (
        ("capture", ORIGINAL, "capture", 0),
        ("default", provider, "compare", 0),
        ("candidate", provider, "compare", 1),
    ):
        for name, digest in captured.items():
            assert sha(Path(name)) == digest, name
        prefix = ROOT / f"int8-lm-head-full-a-v2-native-{phase}-gpu7"
        commands = [f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}",
                    f"export {FLAGS} {FLAG}={selected}"]
        for row in fixtures:
            fixture = Path(row["path"])
            reference = REFERENCES / (fixture.stem + ".reference.bin")
            if phase == "capture" and reference.exists():
                raise FileExistsError(reference)
            profile = ROOT / (prefix.name + "-" + fixture.stem + "-profile")
            commands.append(shlex.join([workspace(PROOF / "independent_compare"), mode,
                workspace(library), workspace(fixture), workspace(reference),
                "/evidence/webgpu-perf-oct20/" + profile.name]))
        execution.write_text(json.dumps(dict(status="running", records=records, phase=phase), indent=2) + "\n")
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                        "--format=csv,noheader"], check=True)
        code, log = run_guard("; ".join(commands), prefix)
        records.append(dict(phase=phase, exit_code=code, prefix=str(prefix)))
        evidence.extend([prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")])
        if code == 2:
            status = f"held_int8_head_{phase}_original_provider_bitwise_mismatch"
            break
        if code != 0 or "FAIL" in log:
            raise RuntimeError("Incomplete independent INT8 arithmetic phase")
        summaries = [line for line in log.splitlines() if line.startswith("INT8_HEAD ")]
        profiles = [host_path(line.removeprefix("PROFILE ")) for line in log.splitlines()
                    if line.startswith("PROFILE ")]
        assert len(summaries) == len(profiles) == len(fixtures)
        phase_coverage = []
        for row, summary, path in zip(fixtures, summaries, profiles):
            assert f"comparisons={row['m'] * row['n'] * 64} mismatches=0" in summary
            counts = collections.Counter(event["name"].rsplit("&", 1)[-1]
                for event in json.loads(path.read_text()) if event.get("cat") == "Kernel")
            selected_count = 64 if selected and row["eligible"] else 0
            assert counts[NAME] == selected_count, (row, counts)
            assert counts["MatMulNBits"] == 64 - selected_count, (row, counts)
            phase_coverage.append(dict(fixture=row["path"], selected=selected_count, actual_weights=row["actual_weights"]))
        coverage.append(dict(phase=phase, fixtures=phase_coverage))
        evidence.extend(profiles)
        if phase == "capture":
            for row in fixtures:
                path = REFERENCES / (Path(row["path"]).stem + ".reference.bin")
                assert path.stat().st_size > 0
                captured[str(path)] = sha(path)
                evidence.append(path)
        print(json.dumps(dict(phase=phase, comparisons=expected, mismatches=0)), flush=True)
    execution.write_text(json.dumps(dict(status=status, records=records), indent=2) + "\n")
    evidence.append(execution)
    outcome.write_text(json.dumps(dict(status=status, records=records, fixtures=len(fixtures),
        comparisons_per_complete_phase=expected, coverage=coverage, captured_sha256=captured,
        independent_original_provider_captures=True, actual_model_int8_weights=True,
        float32_shape_fallbacks_tested=True, all_repeated_outputs_bitwise=True,
        active_model_coverage_verified=False, default_enabled=False, quality_qualified=False,
        source_and_evidence_sha256={str(path): sha(path) for path in evidence}), indent=2) + "\n")
    print(status, flush=True)
