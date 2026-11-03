import json
from pathlib import Path
import shlex
import shutil
import subprocess

from build_qmoe_fc1_double_buffer import FLAG
from build_qmoe_node_replay import sha
from run_hc_narrow_rms_broadcast_native import host_path
from run_dense_columns16_synthetic import run_guard, UUID, workspace


ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-fc1-double-buffer/build-v1")
PROOF = BUILD.parent / "native-proof-v1"
DATA = PROOF / "original-references"
ORIGINAL = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/libonnxruntime_providers_webgpu.so")
FLAGS = (
    "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
    "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_QMOE_COMMAND_REPLAY=1 "
    "ORT_WEBGPU_QMOE_FC1_ACTIVATION=1 ORT_WEBGPU_QMOE_FC2_FINAL_MIX=1 "
    "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE=1 ORT_WEBGPU_QSA_PARALLEL_MAP=1 "
    "ORT_WEBGPU_EXPERT_BITCAST_UNPACK=1 ORT_WEBGPU_DENSE_BITCAST_UNPACK=0 "
    "ORT_WEBGPU_QMOE_INT4_COOPERATIVE=0 ORT_WEBGPU_QMOE_INT4_PREFILL=1 "
    "ORT_WEBGPU_MATMUL_NBITS_SKINNY=0 ORT_WEBGPU_MATMUL_NBITS_HYPER_SKINNY=0"
)


if __name__ == "__main__":
    outcome = ROOT / "qmoe-fc1-double-buffer-v1-native-outcome.json"
    execution = ROOT / "qmoe-fc1-double-buffer-v1-native-execution.json"
    if outcome.exists() or execution.exists() or DATA.exists():
        raise FileExistsError("Existing independent FC1 evidence")
    build_path = BUILD / "build-manifest.json"
    comparator_path = PROOF / "build-manifest.json"
    build, comparator = (json.loads(path.read_text()) for path in (build_path, comparator_path))
    if comparator["status"] != "independent_qmoe_fc1_comparator_built_io_tested":
        raise RuntimeError("Reference comparator not qualified")
    if not (build["baseline_byte_identical"] and build["all_152_original_source_objects_byte_identical"]
            and build["no_object_layout_changes"]):
        raise RuntimeError("Original SDK/object provenance not qualified")
    evidence = [Path(__file__), build_path, comparator_path]
    for name, digest in build["source_sha256"].items():
        path = BUILD / name
        if sha(path) != digest:
            raise RuntimeError(f"Changed FC1 build source: {path}")
        evidence.append(path)
    for name, digest in comparator["source_sha256"].items():
        path = host_path(name)
        if sha(path) != digest:
            raise RuntimeError(f"Changed reference comparator: {path}")
        evidence.append(path)
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    if sha(provider) != build["provider_sha256"] or sha(ORIGINAL) != build["baseline_provider_sha256"]:
        raise RuntimeError("Changed providers")
    parent_path = ROOT / "expert-bitcast-unpack-v2-native-gpu7-20261029.qualification.json"
    parent = json.loads(parent_path.read_text())
    fixtures = [Path("/home/sayanshaw/repos") / name for name in parent["fixture_sha256"]]
    if len(fixtures) != 23 or any(sha(path) != parent["fixture_sha256"][str(path.relative_to("/home/sayanshaw/repos"))]
                                 for path in fixtures):
        raise RuntimeError("Changed inherited QMoE fixtures")
    identity_path = ROOT / "qmoe-hierarchical-gate-v2-native-gpu7-20261029.qualification.json"
    identity = json.loads(identity_path.read_text())
    identity_data = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-hierarchical-gate/fixtures-v1/identity.data")
    if sha(identity_data) != identity["identity_manifest"]["data_sha256"]:
        raise RuntimeError("Changed identity expert weights")
    if shutil.disk_usage(PROOF).free < 1024**3:
        raise RuntimeError("Insufficient space for bounded original captures")
    DATA.mkdir()
    evidence.extend([parent_path, identity_path, identity_data, ORIGINAL, provider, *fixtures])
    records, captured, replay = [], {}, []
    expected = 9748480
    status = "qmoe_fc1_separate_original_outputs_supported"
    for phase, library, mode, selected in (
        ("capture", ORIGINAL, "capture", 0),
        ("default", provider, "compare", 0),
        ("candidate", provider, "compare", 1),
    ):
        for name, digest in captured.items():
            if sha(Path(name)) != digest:
                raise RuntimeError("Changed sealed original output")
        prefix = ROOT / f"qmoe-fc1-double-buffer-v1-native-{phase}-gpu7"
        stats = ROOT / (prefix.name + "-stats.jsonl")
        if stats.exists():
            raise FileExistsError(stats)
        command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
                   f"export {FLAGS} {FLAG}={selected} "
                   f"ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; " +
                   shlex.join([workspace(PROOF / "independent_compare"), workspace(library), mode,
                               workspace(DATA), *[workspace(path) for path in fixtures]]))
        execution.write_text(json.dumps(dict(status="running", records=records, phase=phase), indent=2) + "\n")
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                        "--format=csv,noheader"], check=True)
        code, log = run_guard(command, prefix)
        records.append(dict(phase=phase, exit_code=code, prefix=str(prefix)))
        evidence.extend([prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")])
        if code == 2:
            status = f"held_qmoe_fc1_{phase}_original_provider_bitwise_mismatch"
            break
        if (code != 0 or "FAIL" in log or log.count(" rounds=32 invalid_flag_rejected=1") != 23
                or f"PASS total_exact_elements={expected}; CPU fallback disabled" not in log):
            raise RuntimeError("Incomplete independent FC1 arithmetic batch")
        rows = [json.loads(line) for line in stats.read_text().splitlines()]
        if (len(rows) != 11 or any(row["requests"] != 64 or row["hits"] <= 0
                or row["captures"] + row["hits"] != row["requests"]
                or not 0 < row["entries"] <= row["capacity"] == 8 for row in rows)):
            raise RuntimeError("Required original QMoE command replay not observed")
        evidence.append(stats)
        replay.append(dict(phase=phase, instances=len(rows), requests=sum(row["requests"] for row in rows),
                           hits=sum(row["hits"] for row in rows)))
        if phase == "capture":
            for fixture in fixtures:
                reference = DATA / (fixture.stem + ".reference.bin")
                if reference.stat().st_size <= 0:
                    raise RuntimeError("Empty original reference")
                captured[str(reference)] = sha(reference)
                evidence.append(reference)
        print(json.dumps(dict(phase=phase, comparisons=expected, mismatches=0)), flush=True)
    execution.write_text(json.dumps(dict(status=status, records=records), indent=2) + "\n")
    evidence.append(execution)
    report = dict(status=status, records=records, fixtures=23, rounds_per_fixture=32,
                  comparisons_per_complete_phase=expected, independent_original_provider_captures=True,
                  same_provider_reference_controls_original_fc1_fusion=True,
                  captured_sha256=captured, replay_activity=replay,
                  active_candidate_kernel_coverage_verified=False, model_eligible=False,
                  default_enabled=False, natural_model_timing_performed=False,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    with outcome.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(status, flush=True)
