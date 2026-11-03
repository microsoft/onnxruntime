import json
from pathlib import Path
import shlex
import shutil
import subprocess

from build_hc_parallel16 import FLAG
from build_qmoe_node_replay import sha
from run_dense_columns16_synthetic import workspace

UUID = "GPU-68170cb3-c9ea-a588-7558-ae82635d9173"

def run_guard(command, prefix):
    from run_fp16_dot4_tree_probe import GUARD
    for suffix in (".log", ".guard.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    child = subprocess.run(["python3", str(GUARD), "--gpu-index", "4",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command])
    proof = json.loads(prefix.with_suffix(".guard.json").read_text())
    valid = ((proof["status"] == "completed" and child.returncode == 0) or
             (proof["status"] == "failed" and child.returncode == 2))
    if not valid or proof.get("exit_code") != child.returncode or proof["gpu_uuid"] != UUID:
        raise RuntimeError(f"GPU guard did not prove completed child: {prefix.with_suffix('.guard.json')}")
    return child.returncode, prefix.with_suffix(".log").read_text()



ROOT = Path(__file__).parent
BUILD = Path("/home/sayanshaw/repos/qwen38-int4-hc-parallel16/build-v1")
PROOF = Path("/home/sayanshaw/repos/qwen38-int4-hc-narrow-rms-broadcast/native-proof-v1")
DATA = BUILD.parent / "original-references-attempt3"
ORIGINAL = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/libonnxruntime_providers_webgpu.so")
FLAGS = (
    "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
    "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES=8192 ORT_WEBGPU_STATIC_NODE_REPLAY=1 "
    "ORT_WEBGPU_HC_POST_NORM_PROJECTION=1 ORT_WEBGPU_HC_PROJECTION_SILU=1 "
    "ORT_WEBGPU_QMOE_COMMAND_REPLAY=0 ORT_WEBGPU_QSA_PARALLEL_MAP=1 "
    "ORT_WEBGPU_QMOE_EXTENDED_PREFILL=1 ORT_WEBGPU_DENSE_BITCAST_UNPACK=1"
)


def host_path(path):
    for container, host in (("/workspace/", "/home/sayanshaw/repos/"),
                            ("/evidence/webgpu-perf-oct20/", str(ROOT) + "/")):
        if path.startswith(container):
            return Path(host + path[len(container):])
    raise RuntimeError(f"Unexpected comparator evidence path: {path}")


if __name__ == "__main__":
    outcome = ROOT / "hc-parallel16-v1-native-attempt3-outcome.json"
    execution = ROOT / "hc-parallel16-v1-native-attempt3-execution.json"
    if outcome.exists() or execution.exists() or DATA.exists():
        raise FileExistsError("Existing independent HC evidence")
    build_path = BUILD / "build-manifest.json"
    comparator_path = PROOF / "build-manifest.json"
    build = json.loads(build_path.read_text())
    comparator = json.loads(comparator_path.read_text())
    if comparator["status"] != "independent_hc_comparator_built_io_tested":
        raise RuntimeError("Reference comparator not qualified")
    if not (build["baseline_byte_identical"] and build["all_152_original_source_objects_byte_identical"]
            and build["no_object_layout_changes"]):
        raise RuntimeError("Original SDK/object provenance not qualified")
    evidence = [Path(__file__), ROOT / "prepare_hc_parallel16_native.py",
                ROOT / "build_hc_parallel16.py", ROOT / "test_hc_parallel16.py",
                ROOT / "run_dense_columns16_synthetic.py",
                ROOT / "shared_activation_guard.py", ROOT / "nvml_monitor.py",
                build_path, comparator_path]
    for name, digest in build["source_sha256"].items():
        path = BUILD / name
        if sha(path) != digest:
            raise RuntimeError(f"Changed HC build source: {path}")
        evidence.append(path)
    for name, digest in comparator["source_sha256"].items():
        path = host_path(name)
        if sha(path) != digest:
            raise RuntimeError(f"Changed reference comparator: {path}")
        evidence.append(path)
    provider = BUILD / "libonnxruntime_providers_webgpu.so"
    if sha(provider) != build["provider_sha256"] or sha(ORIGINAL) != build["baseline_provider_sha256"]:
        raise RuntimeError("Changed providers")
    parent_path = ROOT / "hc-post-norm-projection-v2-native-gpu7-20261029.qualification.json"
    parent = json.loads(parent_path.read_text())
    fixtures = sorted(Path("/home/sayanshaw/repos/qwen38-int4-hc-post-norm-projection/fixtures-v1").glob("*.onnx"))
    if len(fixtures) != 36 or any(sha(path) != parent["fixture_sha256"][path.name] for path in fixtures):
        raise RuntimeError("Changed inherited HC fixtures")
    if shutil.disk_usage(PROOF).free < 3 * 1024**3:
        raise RuntimeError("Insufficient space for bounded original captures")
    DATA.mkdir()
    evidence.extend([parent_path, ORIGINAL, provider, *fixtures])
    records, captured, replay = [], {}, []
    status = "hc_separate_original_outputs_supported"
    expected = 12 * 19 * (2 * 10240 + 324) * 128
    for phase, library, mode, selected in (
        ("capture", ORIGINAL, "capture", 0),
        ("default", provider, "compare", 0),
        ("candidate", provider, "compare", 1),
    ):
        for name, digest in captured.items():
            if sha(Path(name)) != digest:
                raise RuntimeError("Changed sealed original output")
        prefix = ROOT / f"hc-parallel16-v1-native-attempt3-{phase}-gpu4"
        stats = ROOT / (prefix.name + "-stats.jsonl")
        if stats.exists():
            raise FileExistsError(stats)
        args = [PROOF / "independent_compare", library]
        command = (f"set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe {UUID}; "
                   f"export {FLAGS} {FLAG}={selected} "
                   f"ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE=/evidence/webgpu-perf-oct20/{stats.name}; " +
                   shlex.join([workspace(args[0]), workspace(args[1]), mode, workspace(DATA),
                               *[workspace(path) for path in fixtures]]))
        execution.write_text(json.dumps(dict(status="running", records=records, phase=phase), indent=2) + "\n")
        subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                        "--format=csv,noheader"], check=True)
        code, log = run_guard(command, prefix)
        records.append(dict(phase=phase, exit_code=code, prefix=str(prefix)))
        evidence.extend([prefix.with_suffix(".log"), prefix.with_suffix(".guard.json")])
        if code == 2:
            status = f"held_hc_{phase}_original_provider_bitwise_mismatch"
            break
        if (code != 0 or "FAIL" in log or log.count("changing_rounds=128 invalid_flag_rejected=1") != 36
                or f"PASS total_exact_elements={expected}; CPU fallback disabled" not in log):
            raise RuntimeError("Incomplete independent HC arithmetic batch")
        rows = [json.loads(line) for line in stats.read_text().splitlines()]
        if (len(rows) != 12
                or sum(row["operator"] == "HCPostNormProjectionF16" for row in rows) != 6
                or sum(row["operator"] == "HCSharedPostNormProjectionF16" for row in rows) != 6
                or any(
                row["requests"] != 96 or row["hits"] <= 0 or row["binding_hits"] <= 0
                or row["rebinds"] + row["binding_hits"] != row["hits"]
                or row["captures"] + row["hits"] != row["requests"]
                or row["evictions"] <= 0 or not 0 < row["entries"] <= row["capacity"] == 64
                or not 0 < row["binding_entries"] <= row["binding_capacity"] == 64
                for row in rows)):
            raise RuntimeError("Required HC replay/binding/LRU activity not observed")
        replay.append(dict(phase=phase, instances=len(rows),
                           requests=sum(row["requests"] for row in rows),
                           hits=sum(row["hits"] for row in rows),
                           binding_hits=sum(row["binding_hits"] for row in rows)))
        evidence.append(stats)
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
    report = dict(status=status, records=records, fixtures=36, rounds_per_fixture=128,
                  comparisons_per_complete_phase=expected, all_four_outputs_compared=True,
                  independent_original_provider_captures=True, captured_sha256=captured,
                  replay_activity=replay,
                  active_candidate_kernel_coverage_verified=False, model_eligible=False,
                  default_enabled=False, natural_model_timing_performed=False,
                  source_and_evidence_sha256={str(path): sha(path) for path in evidence})
    outcome.write_text(json.dumps(report, indent=2) + "\n")
    print(status, flush=True)
