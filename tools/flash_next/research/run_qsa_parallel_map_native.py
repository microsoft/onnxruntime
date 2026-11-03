import argparse
import collections
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha


def packed_score_eligible(events, minimum_stride=512, maximum_heads=16,
                          program_name="SparseAttentionIndexerQsaScore"):
    scores = [e for e in events if e.get("cat") == "Kernel" and
              e["name"].split("&")[-1] == program_name]
    if not scores:
        return False
    assert len(scores) == 18
    eligibility = []
    for event in scores:
        shapes = event["args"]["shapes"]
        query = re.search(r"inputs\[0\] = \{(\d+),(\d+),(\d+)\}", shapes)
        cache = re.search(r"outputs\[2\] = \{(\d+),(\d+)\}", shapes)
        assert query and cache, shapes
        heads = int(event["args"]["cache_key"].split("[", 1)[1].split("|")[1])
        eligibility.append(int(query[1]) * int(query[2]) == 1 and
                           heads <= maximum_heads and int(cache[2]) >= minimum_stride)
    assert len(set(eligibility)) == 1
    return eligibility[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-existing", action="store_true")
    candidates = parser.add_mutually_exclusive_group()
    candidates.add_argument("--combined-map", action="store_true")
    candidates.add_argument("--concat-vector", action="store_true")
    candidates.add_argument("--packed-score", action="store_true")
    parser.add_argument("--version", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.version != 1 and not args.packed_score:
        parser.error("--version is only supported for --packed-score")
    repos = Path("/home/sayanshaw/repos")
    build_dir = repos / (f"qwen38-int4-qsa-packed-score/build-v{args.version}" if args.packed_score else
                         "qwen38-int4-qsa-concat-vector/build-v1" if args.concat_vector else
                         "qwen38-int4-qmoe-fusions-map/build-v2" if args.combined_map
                         else "qwen38-int4-qsa-parallel-map/build-v1")
    prefix = ROOT / (f"qsa-packed-score-v{args.version}-native-gpu7-20261029" if args.packed_score else
                     "qsa-concat-vector-v1-native-gpu7-20261029" if args.concat_vector else
                     "qmoe-fusions-map-v2-native-qsa-gpu7-20261029" if args.combined_map
                     else "qsa-parallel-map-v1-native-original-harness-gpu0-20261029")
    gpu_index = 7 if args.combined_map or args.concat_vector or args.packed_score else 0
    gpu_uuid = ("GPU-0e70e969-bbfc-80c5-d582-88be134632ea" if gpu_index == 7
                else "GPU-c1593c4e-0f5f-7bf7-3567-6c3c19703afd")
    flag = ("ORT_WEBGPU_QSA_PACKED_SCORE" if args.packed_score else
            "ORT_WEBGPU_QSA_CONCAT_VECTOR" if args.concat_vector else "ORT_WEBGPU_QSA_PARALLEL_MAP")
    comparator = repos / "onnxruntime-webgpu-h200-perf/build-int4-qsa-rank/qsa_rank_compare"
    assert args.collect_existing or not prefix.with_suffix(".guard.json").exists()
    assert not prefix.with_suffix(".qualification.json").exists()
    fixtures = sorted(p for directory in (
        ROOT / "qsa-parallel-fixtures",
        repos / "onnxruntime-webgpu-h200-perf/build-int4-qsa-rank-fused-fill/edge-fixtures")
        for p in directory.glob("*.onnx") if not p.stem.endswith("_reference"))
    assert len(fixtures) == 69
    build = json.loads((build_dir / "build-manifest.json").read_text())
    assert sha(build_dir / "libonnxruntime_providers_webgpu.so") == build["provider_sha256"]

    def mounted(path):
        return str(path).replace(str(repos), "/workspace").replace(
            str(ROOT), "/evidence/webgpu-perf-oct20")

    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe "
        f"{gpu_uuid}; export "
        "ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_QSA_PARALLEL_RANK=1 ORT_WEBGPU_QSA_DENSE_MAP=1 "
        "ORT_TEST_QSA_RANK_COMPARE=1 ORT_TEST_QSA_CANDIDATE_ONLY=0; "
        f"for mode in 0 1; do export {flag}=$mode "
        f"ORT_TEST_PROFILE_PREFIX=/evidence/webgpu-perf-oct20/{prefix.name}-profile$mode; " +
        shlex.join([mounted(comparator),
                    mounted(build_dir / "libonnxruntime_providers_webgpu.so"),
                    *(mounted(p) for p in fixtures)]) + "; done")
    invalid_log = prefix.with_name(prefix.name + "-invalid.log")
    if args.concat_vector or args.packed_score:
        command += (
            f"; if {flag}=2 " + shlex.join([mounted(comparator),
                mounted(build_dir / "libonnxruntime_providers_webgpu.so"), mounted(fixtures[0])]) +
            f" > {shlex.quote(mounted(invalid_log))} 2>&1; then exit 1; fi; " +
            shlex.join(["grep", "-Fq", f"{flag} must be 0 or 1", mounted(invalid_log)]))
    if not args.collect_existing:
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", str(gpu_index),
            "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "600",
            "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert guard["gpu_index"] == gpu_index and guard["gpu_uuid"] == gpu_uuid
    assert guard["command"] == ["bash", "-lc", command]
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count(" exact_checks=18 candidate_only=0 rank_compare=1 validation=full") == 138
    profiles = list(ROOT.glob(prefix.name + "-profile*-*.json"))
    if args.concat_vector or args.packed_score:
        negative_programs = {(), ("SparseAttentionIndexerQsaConcat",)} if args.packed_score else {()}
        rejected_profiles = [p for p in profiles if tuple(
            e["name"].split("&")[-1] for e in json.loads(p.read_text())
            if e.get("cat") == "Kernel") in negative_programs]
        assert len(rejected_profiles) == 1, rejected_profiles
        assert rejected_profiles[0].name.startswith(prefix.name + "-profile1-" + fixtures[0].stem + "_")
        assert f"{flag} must be 0 or 1" in invalid_log.read_text()
        profiles = [p for p in profiles if p not in rejected_profiles]
    assert len(profiles) == 138, len(profiles)
    exercised = []
    packed_expected = set()
    minimum_score_stride = 2048 if args.version == 2 else 512
    for path in profiles:
        events = json.loads(path.read_text())
        if args.packed_score and "-profile0-" in path.name and packed_score_eligible(
                events, minimum_stride=minimum_score_stride):
            packed_expected.add(path.name.split("-profile0-", 1)[1].rsplit("_2026-", 1)[0])
        programs = collections.defaultdict(list)
        for event in events:
            if event.get("cat") == "Kernel":
                programs[event["name"].split("&")[-1]].append(event["dur"])
        count = programs["QsaPackedScore4" if args.packed_score else
                         "QsaConcatVector4" if args.concat_vector else "QsaParallelMapCount"]
        scatter = count if args.concat_vector or args.packed_score else programs["QsaParallelMapScatter"]
        assert len(count) == len(scatter)
        if count:
            assert len(count) == 18 and "-profile1-" in path.name
            baseline_path = Path(str(path).replace("-profile1-", "-profile0-"))
            matches = list(ROOT.glob(baseline_path.name.rsplit("_2026-", 1)[0] + "_2026-*.json"))
            assert len(matches) == 1
            old_maps = [e["dur"] for e in json.loads(matches[0].read_text())
                        if e.get("cat") == "Kernel" and
                        e["name"].split("&")[-1] == (
                            "SparseAttentionIndexerQsaScore" if args.packed_score else
                            "SparseAttentionIndexerQsaConcat" if args.concat_vector
                            else "SparseAttentionIndexerQsaMap")]
            assert len(old_maps) == 18
            timings = (dict(original_score_median_us=statistics.median(old_maps),
                            packed_score_median_us=statistics.median(count)) if args.packed_score else
                       dict(original_concat_median_us=statistics.median(old_maps),
                            vector_concat_median_us=statistics.median(count)) if args.concat_vector else
                       dict(original_map_median_us=statistics.median(old_maps),
                            count_median_us=statistics.median(count), scatter_median_us=statistics.median(scatter)))
            exercised.append(dict(profile=path.name, executions=len(count), **timings))
    expected = {
        "qsa_past8192_rows1_ratio4_half1_packed1_h4_d128_budget2048_model_rotary",
        "qsa_past8192_rows1_ratio4_half1_packed1_h4_d128_budget2052_model_irregular_budget",
        "qsa_past32768_rows1_ratio4_half1_packed1_h4_d128_budget2048_model_rotary",
        *(f"qsa_rank_edge_past{past}_half{half}_ties"
          for past in (33024, 65535) for half in (0, 1)),
    }
    actual = {row["profile"].split("-profile1-", 1)[1].rsplit("_2026-", 1)[0] for row in exercised}
    if args.concat_vector:
        expected = {p.stem for p in fixtures}
        assert f"{flag} must be 0 or 1" in invalid_log.read_text()
    if args.packed_score:
        assert packed_expected
        expected = packed_expected
    assert actual == expected, actual
    report = dict(status=("exact_reference_and_packed_score_dispatch_qualified" if args.packed_score else
                          "exact_reference_and_vector_concat_dispatch_qualified" if args.concat_vector
                          else "exact_reference_and_parallel_map_dispatch_qualified"),
        fixtures=69, reference_executions=69 * 36, both_flags_tested=True,
        cpu_fallback_disabled=True, graph_optimizations_disabled=True,
        provider_sha256=build["provider_sha256"],
        baseline_provider_sha256=build["baseline_provider_sha256"],
        combined_qmoe_fusions=args.combined_map,
        concat_vector=args.concat_vector,
        packed_score=args.packed_score,
        minimum_score_stride=minimum_score_stride if args.packed_score else None,
        comparator_sha256=sha(comparator), comparator=str(comparator),
        comparator_scope=f"Original previously qualified QSA rank harness and runtime; outer {flag}=0/1, inner rank0/rank1",
        fixture_sha256={str(p): sha(p) for p in fixtures}, exercised=exercised,
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "fixtures", "reference_executions")}))
    print(json.dumps(exercised, indent=2))


if __name__ == "__main__":
    main()
