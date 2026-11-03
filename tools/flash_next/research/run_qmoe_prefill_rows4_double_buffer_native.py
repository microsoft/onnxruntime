import collections
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from build_qmoe_prefill_rows4_double_buffer import FLAG, NAME
from build_qmoe_node_replay import sha
from prepare_qmoe_extended_prefill import DIRECTORY
from run_dense_columns16_synthetic import run_guard, UUID, workspace
from run_hc_narrow_rms_broadcast_native import host_path
from run_qmoe_prefill256_native import verify_gate_coverage


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4-double-buffer/build-v2")
PROOF = BUILD.parent / "native-proof-v2"
DATA = PROOF / "original-references"
ORIGINAL = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/libonnxruntime_providers_webgpu.so")
FLAGS = (
    "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
    "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_QMOE_COMMAND_REPLAY=1 "
    "ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 ORT_WEBGPU_QMOE_INT4_PREFILL=1 "
    "ORT_WEBGPU_QMOE_EXTENDED_PREFILL=1 ORT_WEBGPU_QMOE_PREFILL256=1 "
    "ORT_WEBGPU_QMOE_PREFILL512=1 ORT_WEBGPU_QMOE_PREFILL1024=1 "
    "ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1 "
    "ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0"
)


if __name__ == "__main__":
    outcome = ROOT / "qmoe-prefill-rows4-double-buffer-v2-native-outcome.json"
    execution = ROOT / "qmoe-prefill-rows4-double-buffer-v2-native-execution.json"
    if outcome.exists() or execution.exists() or DATA.exists():
        raise FileExistsError("Existing independent prefill evidence")
    build_path, comparator_path = BUILD / "build-manifest.json", PROOF / "build-manifest.json"
    build, comparator = (json.loads(path.read_text()) for path in (build_path, comparator_path))
    assert comparator["status"] == "independent_prefill_comparator_built_io_tested"
    assert build["baseline_byte_identical"] and build["all_152_original_source_objects_byte_identical"]
    assert build["original_qmoe_translation_unit_sha256"] and build["qmoe_replay_selector_keyed"]
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
    parent_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
    parent = json.loads(parent_path.read_text())
    assert parent["provider_sha256"] == build["baseline_provider_sha256"]
    fixture_manifest = DIRECTORY / "manifest.json"
    manifest = json.loads(fixture_manifest.read_text())
    assert manifest == parent["fixture_manifest"]
    assert Path(manifest["source_weights"]).samefile(DIRECTORY / "model.onnx.data")
    fixtures = sorted(DIRECTORY.glob("*.onnx"))
    assert len(fixtures) == 6
    for path in fixtures:
        record = manifest["fixtures"][path.name]
        assert sha(path) == record["fixture_sha256"]
        assert sha(Path(record["source"])) == record["source_sha256"]
    evidence.extend([parent_path, fixture_manifest, *fixtures])
    if shutil.disk_usage(PROOF).free < 2 * 1024**3:
        raise RuntimeError("Insufficient bounded original-reference storage")
    DATA.mkdir()
    expected = parent["finite_bitwise_comparisons"]
    assert expected == 300810240
    records, captured, coverage = [], {}, []
    status = "prefill_separate_original_outputs_supported"
    for phase, library, mode, selected in (
        ("capture", ORIGINAL, "capture", 0),
        ("default", provider, "compare", 0),
        ("candidate", provider, "compare", 1),
    ):
        for name, digest in captured.items():
            assert sha(Path(name)) == digest, name
        prefix = ROOT / f"qmoe-prefill-rows4-double-buffer-v2-native-{phase}-gpu7"
        command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
            f"export {FLAGS} {FLAG}={selected}; " +
            shlex.join([workspace(PROOF / "independent_compare"), workspace(library), mode,
                        workspace(DATA), *[workspace(path) for path in fixtures]]))
        execution.write_text(json.dumps(dict(status="running", records=records, phase=phase), indent=2) + "\n")
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                        "--format=csv,noheader"], check=True)
        code, log = run_guard(command, prefix)
        records.append(dict(phase=phase, exit_code=code, prefix=str(prefix)))
        evidence.extend([prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")])
        if code == 2:
            status = f"held_prefill_{phase}_original_provider_bitwise_mismatch"
            break
        if (code != 0 or "FAIL" in log or log.count("retained_chunk16_reference=1 invalid_flag_rejected=1") != 6
                or f"PASS total_exact_elements={expected}; CPU fallback disabled" not in log):
            raise RuntimeError("Incomplete independent prefill arithmetic batch")
        profiles = [host_path(line.removeprefix("PROFILE ")) for line in log.splitlines()
                    if line.startswith("PROFILE ")]
        assert len(profiles) == 6
        phase_coverage = []
        for path in profiles:
            events = json.loads(path.read_text())
            verify_gate_coverage(events, {16: 976, 128: 8, 256: 8, 512: 8, 1024: 8})
            counts = collections.Counter(event["name"].rsplit("&", 1)[-1] for event in events
                                         if event.get("cat") == "Kernel")
            assert counts[NAME] == (16 if selected else 0), counts
            assert counts["MatMulNBitsGroupedRows4F16"] == (16 if selected else 32), counts
            phase_coverage.append(dict(profile=str(path), selected=counts[NAME],
                                       original=counts["MatMulNBitsGroupedRows4F16"]))
        coverage.append(dict(phase=phase, fixtures=phase_coverage))
        evidence.extend(profiles)
        if phase == "capture":
            for fixture in fixtures:
                reference = DATA / (fixture.stem + ".reference.bin")
                assert reference.stat().st_size > 0
                captured[str(reference)] = sha(reference)
                evidence.append(reference)
        print(json.dumps(dict(phase=phase, comparisons=expected, mismatches=0)), flush=True)
    execution.write_text(json.dumps(dict(status=status, records=records), indent=2) + "\n")
    evidence.append(execution)
    outcome.write_text(json.dumps(dict(
        status=status, records=records, comparisons_per_complete_phase=expected,
        fixtures=6, rows_tested=parent["rows_tested"], pending_windows=[1, 64],
        independent_original_provider_captures=True, captured_sha256=captured, coverage=coverage,
        model_eligible=False, default_enabled=False,
        source_and_evidence_sha256={str(path): sha(path) for path in evidence}), indent=2) + "\n")
    print(status, flush=True)
