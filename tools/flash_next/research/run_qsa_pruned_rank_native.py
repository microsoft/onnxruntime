import json
from pathlib import Path
import shlex
import statistics
import subprocess
import re

from build_qsa_pruned_rank import FLAG
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_qsa_indexer_stack_native import REPOS, UUID, programs


def rank_stride_eligible(events, minimum_stride):
    ranks = [e for e in events if e.get("cat") == "Kernel" and
             e["name"].endswith("&SparseAttentionIndexerQsaParallelRank")]
    if not ranks:
        return False
    shapes = [re.search(r"inputs\[1\] = \{(\d+),(\d+)\}", e["args"]["shapes"]) for e in ranks]
    assert all(shapes), "Missing original rank score-cache dimensions"
    return all(int(shape[2]) >= minimum_stride for shape in shapes)


def main(variant="qsa-pruned-rank-v1", selector=FLAG, program_name="QsaPrunedRank",
         report_status="exact_reference_and_pruned_rank_dispatch_qualified",
         workgroup_size=32, pruning_enabled=True, requires_subgroups=True,
         parent_path=ROOT / "dense-block32-v2-contexts-20261030.qualification.json",
         stamp="20261030", auxiliary_programs=(), minimum_score_stride=None):
    family, version = variant.rsplit("-v", 1)
    directory = REPOS / f"qwen38-int4-{family}/build-v{version}"
    prefix = ROOT / f"{variant}-native-gpu7-{stamp}"
    assert not any(prefix.with_suffix(s).exists() for s in (".qualification.json", ".guard.json"))
    build = json.loads((directory / "build-manifest.json").read_text())
    parent = json.loads(parent_path.read_text())
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == parent["provider_sha256"]
    provider = directory / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == build["provider_sha256"]
    frozen = json.loads((ROOT / "qsa-score16-v5-native-gpu7-20261029.qualification.json").read_text())
    fixtures = sorted(Path(p) for p in frozen["fixture_sha256"])
    assert len(fixtures) == 69
    for path, digest in {**frozen["fixture_sha256"], **frozen["reference_sha256"]}.items():
        assert sha(Path(path)) == digest
    comparator = REPOS / "onnxruntime-webgpu-h200-perf/build-int4-qsa-rank/qsa_rank_compare"
    assert sha(comparator) == frozen["comparator_sha256"]

    def mounted(path):
        return str(path).replace(str(REPOS), "/workspace").replace(str(ROOT), "/evidence/webgpu-perf-oct20")

    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID +
        "; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_QSA_PARALLEL_RANK=1 ORT_WEBGPU_QSA_DENSE_MAP=1 "
        "ORT_WEBGPU_QSA_INDEXER_STACK=1 ORT_WEBGPU_QSA_SCORE16=1 "
        "ORT_WEBGPU_QSA_CONCAT_VECTOR=0 ORT_WEBGPU_QSA_PACKED_SCORE=0 "
        "ORT_TEST_QSA_RANK_COMPARE=1 ORT_TEST_QSA_CANDIDATE_ONLY=0; "
        f"for mode in 0 1; do export {selector}=$mode "
        f"ORT_TEST_PROFILE_PREFIX=/evidence/webgpu-perf-oct20/{prefix.name}-profile$mode; " +
        shlex.join([mounted(comparator), mounted(provider), *(mounted(p) for p in fixtures)]) + "; done")
    invalid = prefix.with_name(prefix.name + "-invalid.log")
    command += (
        f"; if {selector}=2 ORT_TEST_PROFILE_PREFIX={shlex.quote(mounted(prefix) + '-invalid-profile')} " +
        shlex.join([mounted(comparator), mounted(provider), mounted(fixtures[0])]) +
        f" > {shlex.quote(mounted(invalid))} 2>&1; then exit 1; fi; " +
        shlex.join(["grep", "-Fq", f"{selector} must be 0 or 1", mounted(invalid)]))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "1200",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0 and guard["gpu_index"] == 7
    assert guard["gpu_uuid"] == UUID and guard["command"] == ["bash", "-lc", command]
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count(" exact_checks=18 candidate_only=0 rank_compare=1 validation=full") == 138
    sources = {}
    exercised = []
    for fixture in fixtures:
        events = []
        for mode in (0, 1):
            paths = list(ROOT.glob(f"{prefix.name}-profile{mode}-{fixture.stem}_2026-*.json"))
            assert len(paths) == 1, paths
            events.append(json.loads(paths[0].read_text()))
            sources[paths[0].name] = sha(paths[0])
        before, after = (programs(e) for e in events)
        assert before[program_name] == 0
        admitted = minimum_score_stride is None or rank_stride_eligible(events[0], minimum_score_stride)
        if admitted:
            assert after["SparseAttentionIndexerQsaParallelRank"] == 0
            assert after[program_name] == before["SparseAttentionIndexerQsaParallelRank"]
        else:
            assert after[program_name] == 0
            assert before["SparseAttentionIndexerQsaParallelRank"] == after["SparseAttentionIndexerQsaParallelRank"]
        for name in auxiliary_programs:
            assert before[name] == 0 and after[name] == after[program_name]
        for name in set(before) | set(after):
            if name not in (program_name, "SparseAttentionIndexerQsaParallelRank", *auxiliary_programs):
                assert before[name] == after[name], (fixture, name)
        if after[program_name]:
            durations = [statistics.median(e["dur"] for e in data if e.get("cat") == "Kernel" and
                         e["name"].endswith("&" + name))
                         for data, name in zip(events, ("SparseAttentionIndexerQsaParallelRank", program_name))]
            durations[1] += sum(statistics.median(e["dur"] for e in events[1] if e.get("cat") == "Kernel"
                                                and e["name"].endswith("&" + name))
                                for name in auxiliary_programs)
            exercised.append(dict(fixture=fixture.name, dispatches=after[program_name],
                                  original_median_us=durations[0], candidate_median_us=durations[1]))
    assert exercised and len(sources) == 138
    assert f"{selector} must be 0 or 1" in invalid.read_text()
    negatives = list(ROOT.glob(prefix.name + "-invalid-profile-*.json"))
    assert len(negatives) == 1 and not programs(json.loads(negatives[0].read_text()))
    for path in (negatives[0], invalid, prefix.with_suffix(".guard.json"), prefix.with_suffix(".log")):
        sources[path.name] = sha(path)
    report = dict(status=report_status,
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        fixtures=69, reference_executions=2484, exercised=exercised,
        invalid_selector_rejected=True, cpu_fallback_disabled=True, graph_optimizations_disabled=True,
        validation="full", original_float_scoring_and_nan_tie_tail_semantics_preserved=True,
        fixed_subgroup_size=32 if requires_subgroups else None, requires_subgroups=requires_subgroups,
        candidate_zero_complete_valid_count=True, auxiliary_programs=list(auxiliary_programs),
        minimum_score_stride=minimum_score_stride,
        workgroup_size=workgroup_size, pruning_enabled=pruning_enabled, program_name=program_name,
        build_manifest_sha256=sha(directory / "build-manifest.json"),
        comparator_sha256=sha(comparator), fixture_sha256=frozen["fixture_sha256"],
        reference_sha256=frozen["reference_sha256"], source_sha256=sources,
        instrumented_not_a_speed_claim=True, quality_qualified=False, headline_eligible=False)
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "reference_executions", "exercised")}), flush=True)


if __name__ == "__main__":
    main()
