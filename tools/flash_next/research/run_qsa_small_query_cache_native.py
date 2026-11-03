import argparse
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess

from build_qsa_small_query_cache import FLAG, PROGRAM
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_qsa_indexer_stack_native import REPOS, UUID, programs
from run_qsa_parallel_map_native import packed_score_eligible


def original_score(events):
    names = programs(events)
    return next((name for name in ("QsaPackedScore4", "SparseAttentionIndexerQsaScore") if names[name]), None)


def eligible(events):
    name = original_score(events)
    if name is None:
        return False
    scores = [event for event in events if event.get("cat") == "Kernel" and
              event["name"].split("&")[-1] == name]
    if not all(re.search(r"outputs\[2\] = \{(\d+),(\d+)\}", event["args"]["shapes"]) for event in scores):
        return False
    return (packed_score_eligible(events, minimum_stride=128, maximum_heads=4,
                                  program_name=name) and
            not packed_score_eligible(events, minimum_stride=2048, maximum_heads=4,
                                      program_name=name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-existing", action="store_true")
    args = parser.parse_args()
    directory = REPOS / "qwen38-int4-qsa-small-query-cache/build-v2"
    prefix = ROOT / "qsa-small-query-cache-v2-native-gpu7-20261031"
    assert not prefix.with_suffix(".qualification.json").exists()
    assert args.collect_existing or not prefix.with_suffix(".guard.json").exists()
    build = json.loads((directory / "build-manifest.json").read_text())
    provider = directory / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == build["provider_sha256"]
    parent = json.loads((ROOT / "qsa-prefill-score16-v1-contexts-20261031.qualification.json").read_text())
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == parent["provider_sha256"]
    original = json.loads((ROOT / "qsa-indexer-stack-v1-native-gpu7-20261029.qualification.json").read_text())
    fixtures = sorted(Path(path) for path in original["fixture_sha256"])
    assert len(fixtures) == 69
    for path in fixtures:
        assert sha(path) == original["fixture_sha256"][str(path)]
    for path, digest in original["reference_sha256"].items():
        assert sha(Path(path)) == digest
    comparator = REPOS / "onnxruntime-webgpu-h200-perf/build-int4-qsa-rank/qsa_rank_compare"
    assert sha(comparator) == original["comparator_sha256"]

    def mounted(path):
        return str(path).replace(str(REPOS), "/workspace").replace(str(ROOT), "/evidence/webgpu-perf-oct20")

    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID +
        "; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_QSA_PARALLEL_RANK=1 ORT_WEBGPU_QSA_DENSE_MAP=1 "
        "ORT_WEBGPU_QSA_INDEXER_STACK=1 ORT_WEBGPU_QSA_CONCAT_VECTOR=0 ORT_WEBGPU_QSA_PACKED_SCORE=0 "
        "ORT_WEBGPU_QSA_SCORE16=1 ORT_WEBGPU_QSA_PREFILL_SCORE16=1 "
        "ORT_TEST_QSA_RANK_COMPARE=1 ORT_TEST_QSA_CANDIDATE_ONLY=0; "
        f"for mode in 0 1; do export {FLAG}=$mode "
        f"ORT_TEST_PROFILE_PREFIX=/evidence/webgpu-perf-oct20/{prefix.name}-profile$mode; " +
        shlex.join([mounted(comparator), mounted(provider), *(mounted(path) for path in fixtures)]) + "; done")
    invalid_log = prefix.with_name(prefix.name + "-invalid.log")
    command += (
        f"; if {FLAG}=2 ORT_TEST_PROFILE_PREFIX={shlex.quote(mounted(prefix) + '-invalid-profile')} " +
        shlex.join([mounted(comparator), mounted(provider), mounted(fixtures[0])]) +
        f" > {shlex.quote(mounted(invalid_log))} 2>&1; then exit 1; fi; " +
        shlex.join(["grep", "-Fq", f"{FLAG} must be 0 or 1", mounted(invalid_log)]))
    if not args.collect_existing:
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "1200",
                        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert guard["gpu_index"] == 7 and guard["gpu_uuid"] == UUID
    assert guard["command"] == ["bash", "-lc", command]
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count(" exact_checks=18 candidate_only=0 rank_compare=1 validation=full") == 138
    assert len(list(ROOT.glob(prefix.name + "-profile[0-1]-*.json"))) == 138
    exercised, sources = [], {}
    for fixture in fixtures:
        events = []
        for mode in (0, 1):
            matches = list(ROOT.glob(f"{prefix.name}-profile{mode}-{fixture.stem}_2026-*.json"))
            assert len(matches) == 1, matches
            events.append(json.loads(matches[0].read_text()))
            sources[matches[0].name] = sha(matches[0])
        admitted = eligible(events[0])
        original_name = original_score(events[0])
        before, after = (programs(event) for event in events)
        assert before[PROGRAM] == 0
        assert after[PROGRAM] == (18 if admitted else 0), fixture
        if admitted:
            assert original_name is not None and after[original_name] == 0
        for name in set(before) | set(after):
            if name not in ((original_name, PROGRAM) if admitted else (PROGRAM,)):
                assert before[name] == after[name], (fixture, name)
        if admitted:
            timings = [statistics.median(event["dur"] for event in data if event.get("cat") == "Kernel" and
                                        event["name"].endswith("&" + name))
                       for data, name in zip(events, (original_name, PROGRAM))]
            exercised.append(dict(fixture=fixture.name, original_program=original_name, original_median_us=timings[0],
                                  query_cache_median_us=timings[1]))
    assert exercised
    assert f"{FLAG} must be 0 or 1" in invalid_log.read_text()
    negative = list(ROOT.glob(prefix.name + "-invalid-profile-*.json"))
    assert len(negative) == 1 and not programs(json.loads(negative[0].read_text()))
    for path in (negative[0], invalid_log, prefix.with_suffix(".guard.json"), prefix.with_suffix(".log")):
        sources[path.name] = sha(path)
    sources["qsa-prefill-score16-v1-contexts-20261031.qualification.json"] = sha(
        ROOT / "qsa-prefill-score16-v1-contexts-20261031.qualification.json")
    report = dict(status="exact_reference_and_small_query_cache_dispatch_qualified",
                  provider_sha256=build["provider_sha256"],
                  baseline_provider_sha256=build["baseline_provider_sha256"],
                  fixtures=69, reference_executions=2484, exercised=exercised,
                  score_dispatches=18 * len(exercised), program_name=PROGRAM,
                  minimum_score_stride=128, maximum_score_stride_exclusive=2048,
                  invalid_selector_rejected=True, cpu_fallback_disabled=True,
                  graph_optimizations_disabled=True, validation="full",
                  comparator_sha256=sha(comparator), fixture_sha256=original["fixture_sha256"],
                  reference_sha256=original["reference_sha256"], source_sha256=sources,
                  quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "fixtures", "reference_executions", "score_dispatches", "exercised")}), flush=True)


if __name__ == "__main__":
    main()
