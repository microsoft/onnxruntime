import argparse
import collections
import json
from pathlib import Path
import shlex
import subprocess

from qualify_quickgelu_mul_long_context import ROOT, sha
from run_qsa_parallel_map_native import packed_score_eligible


REPOS = Path("/home/sayanshaw/repos")
UUID = "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
MODES = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1))


def programs(events):
    return collections.Counter(e["name"].split("&")[-1] for e in events if e.get("cat") == "Kernel")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-existing", action="store_true")
    args = parser.parse_args()
    directory = REPOS / "qwen38-int4-qsa-indexer-stack/build-v1"
    prefix = ROOT / "qsa-indexer-stack-v1-native-gpu7-20261029"
    assert not prefix.with_suffix(".qualification.json").exists()
    assert args.collect_existing or not prefix.with_suffix(".guard.json").exists()
    build = json.loads((directory / "build-manifest.json").read_text())
    provider = directory / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == build["provider_sha256"]
    copy = json.loads((ROOT / "qsa-concat-vector-v1-native-gpu7-20261029.qualification.json").read_text())
    score = json.loads((ROOT / "qsa-packed-score-v2-native-gpu7-20261029.qualification.json").read_text())
    assert copy["status"] == "exact_reference_and_vector_concat_dispatch_qualified"
    assert score["status"] == "exact_reference_and_packed_score_dispatch_qualified"
    assert score["minimum_score_stride"] == 2048
    assert copy["baseline_provider_sha256"] == score["baseline_provider_sha256"] == build["baseline_provider_sha256"]
    assert copy["fixture_sha256"] == score["fixture_sha256"]
    fixtures = sorted(Path(p) for p in copy["fixture_sha256"])
    assert len(fixtures) == 69
    references = {}
    for path in fixtures:
        assert sha(path) == copy["fixture_sha256"][str(path)]
        reference = path.with_name(path.stem + "_reference.onnx")
        references[str(reference)] = sha(reference)
    comparator = REPOS / "onnxruntime-webgpu-h200-perf/build-int4-qsa-rank/qsa_rank_compare"
    assert sha(comparator) == copy["comparator_sha256"] == score["comparator_sha256"]

    def mounted(path):
        return str(path).replace(str(REPOS), "/workspace").replace(str(ROOT), "/evidence/webgpu-perf-oct20")

    cases = " ".join(
        f"{mode}) export ORT_WEBGPU_QSA_CONCAT_VECTOR={c} ORT_WEBGPU_QSA_PACKED_SCORE={s} "
        f"ORT_WEBGPU_QSA_INDEXER_STACK={stack};;"
        for mode, (c, s, stack) in enumerate(MODES))
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID +
        "; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_QSA_PARALLEL_RANK=1 ORT_WEBGPU_QSA_DENSE_MAP=1 "
        "ORT_TEST_QSA_RANK_COMPARE=1 ORT_TEST_QSA_CANDIDATE_ONLY=0; "
        'for mode in 0 1 2 3 4; do case "$mode" in ' + cases + " esac; " +
        f"export ORT_TEST_PROFILE_PREFIX=/evidence/webgpu-perf-oct20/{prefix.name}-profile$mode; " +
        shlex.join([mounted(comparator), mounted(provider), *(mounted(p) for p in fixtures)]) + "; done")
    invalid_log = prefix.with_name(prefix.name + "-invalid.log")
    invalid_prefix = mounted(prefix) + "-invalid-profile"
    command += (
        "; if ORT_WEBGPU_QSA_CONCAT_VECTOR=0 ORT_WEBGPU_QSA_PACKED_SCORE=0 "
        f"ORT_WEBGPU_QSA_INDEXER_STACK=2 ORT_TEST_PROFILE_PREFIX={shlex.quote(invalid_prefix)} " +
        shlex.join([mounted(comparator), mounted(provider), mounted(fixtures[0])]) +
        f" > {shlex.quote(mounted(invalid_log))} 2>&1; then exit 1; fi; " +
        shlex.join(["grep", "-Fq", "ORT_WEBGPU_QSA_INDEXER_STACK must be 0 or 1", mounted(invalid_log)]))
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
    assert log.count(" exact_checks=18 candidate_only=0 rank_compare=1 validation=full") == 345
    profiles = list(ROOT.glob(prefix.name + "-profile[0-4]-*.json"))
    assert len(profiles) == 345
    copy_dispatches = score_dispatches = 0
    source_hashes = {}
    for fixture in fixtures:
        events = []
        for mode in range(5):
            matches = list(ROOT.glob(f"{prefix.name}-profile{mode}-{fixture.stem}_2026-*.json"))
            assert len(matches) == 1, matches
            events.append(json.loads(matches[0].read_text()))
            source_hashes[matches[0].name] = sha(matches[0])
        eligible = packed_score_eligible(events[0], minimum_stride=2048)
        baseline = programs(events[0])
        assert baseline["SparseAttentionIndexerQsaConcat"] == 18
        for data, (c, s, stack) in zip(events, MODES):
            actual = programs(data)
            copies = 18 if c or stack else 0
            scores = 18 if eligible and (s or stack) else 0
            assert actual["QsaConcatVector4"] == copies
            assert actual["QsaPackedScore4"] == scores
            assert actual["SparseAttentionIndexerQsaConcat"] == 18 - copies
            assert actual["SparseAttentionIndexerQsaScore"] == (
                0 if scores else baseline["SparseAttentionIndexerQsaScore"])
            copy_dispatches += copies
            score_dispatches += scores
    assert copy_dispatches == 69 * 3 * 18
    assert score_dispatches == len(score["exercised"]) * 3 * 18
    assert "ORT_WEBGPU_QSA_INDEXER_STACK must be 0 or 1" in invalid_log.read_text()
    negative = list(ROOT.glob(prefix.name + "-invalid-profile-*.json"))
    assert len(negative) == 1 and not programs(json.loads(negative[0].read_text()))
    report = dict(status="exact_reference_and_indexer_stack_dispatch_qualified",
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        fixtures=69, reference_executions=6210, modes=MODES,
        copy_dispatches=copy_dispatches, packed_score_dispatches=score_dispatches,
        invalid_selector_rejected=True, cpu_fallback_disabled=True, graph_optimizations_disabled=True,
        validation="full", comparator_sha256=sha(comparator), fixture_sha256=copy["fixture_sha256"],
        reference_sha256=references, profile_sha256=source_hashes,
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "status", "fixtures", "reference_executions", "copy_dispatches", "packed_score_dispatches")}))


if __name__ == "__main__":
    main()
