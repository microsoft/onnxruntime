import argparse
import json
from pathlib import Path
import shlex
import statistics
import subprocess

from build_qsa_score16 import FLAG
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_qsa_indexer_stack_native import REPOS, UUID, programs
from run_qsa_parallel_map_native import packed_score_eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-existing", action="store_true")
    parser.add_argument("--version", type=int, choices=(1, 2, 3, 4, 5), default=2)
    args = parser.parse_args()
    program_name = ("QsaPackedScore16QueryWide" if args.version == 5 else
                    "QsaPackedScore16QueryCache" if args.version == 4 else
                    "QsaPackedScore16Padded" if args.version == 3 else "QsaPackedScore16")
    minimum_stride = 8192 if args.version == 2 else 2048
    directory = REPOS / f"qwen38-int4-qsa-score16/build-v{args.version}"
    prefix = ROOT / f"qsa-score16-v{args.version}-native-gpu7-20261029"
    assert not prefix.with_suffix(".qualification.json").exists()
    assert args.collect_existing or not prefix.with_suffix(".guard.json").exists()
    build = json.loads((directory / "build-manifest.json").read_text())
    provider = directory / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == build["provider_sha256"]
    parent = json.loads((ROOT / "prefill256-v1-contexts-20261029.qualification.json").read_text())
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == parent["provider_sha256"]
    original = json.loads((ROOT / "qsa-indexer-stack-v1-native-gpu7-20261029.qualification.json").read_text())
    fixtures = sorted(Path(p) for p in original["fixture_sha256"])
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
        "ORT_TEST_QSA_RANK_COMPARE=1 ORT_TEST_QSA_CANDIDATE_ONLY=0; "
        f"for mode in 0 1; do export {FLAG}=$mode "
        f"ORT_TEST_PROFILE_PREFIX=/evidence/webgpu-perf-oct20/{prefix.name}-profile$mode; " +
        shlex.join([mounted(comparator), mounted(provider), *(mounted(p) for p in fixtures)]) + "; done")
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
    exercised = []
    sources = {}
    for fixture in fixtures:
        paths = []
        events = []
        for mode in (0, 1):
            matches = list(ROOT.glob(f"{prefix.name}-profile{mode}-{fixture.stem}_2026-*.json"))
            assert len(matches) == 1, matches
            paths.append(matches[0])
            events.append(json.loads(matches[0].read_text()))
            sources[matches[0].name] = sha(matches[0])
        eligible = packed_score_eligible(events[0], minimum_stride=minimum_stride, maximum_heads=4,
                                         program_name="QsaPackedScore4")
        before, after = (programs(e) for e in events)
        assert before[program_name] == 0
        assert after[program_name] == (18 if eligible else 0)
        assert after["QsaPackedScore4"] == (0 if eligible else before["QsaPackedScore4"])
        for name in set(before) | set(after):
            if name not in ("QsaPackedScore4", program_name):
                assert before[name] == after[name], (fixture, name)
        if eligible:
            timings = [statistics.median(e["dur"] for e in data if e.get("cat") == "Kernel" and
                       e["name"].endswith("&" + name)) for data, name in
                       zip(events, ("QsaPackedScore4", program_name))]
            exercised.append(dict(fixture=fixture.name, packed4_median_us=timings[0],
                                  packed16_median_us=timings[1]))
    assert exercised
    assert f"{FLAG} must be 0 or 1" in invalid_log.read_text()
    negative = list(ROOT.glob(prefix.name + "-invalid-profile-*.json"))
    assert len(negative) == 1 and not programs(json.loads(negative[0].read_text()))
    for path in (negative[0], invalid_log, prefix.with_suffix(".guard.json"), prefix.with_suffix(".log")):
        sources[path.name] = sha(path)
    report = dict(status="exact_reference_and_score16_dispatch_qualified",
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        fixtures=69, reference_executions=2484, exercised=exercised,
        minimum_score_stride=minimum_stride, program_name=program_name,
        score16_dispatches=18 * len(exercised), invalid_selector_rejected=True,
        cpu_fallback_disabled=True, graph_optimizations_disabled=True, validation="full",
        comparator_sha256=sha(comparator), fixture_sha256=original["fixture_sha256"],
        reference_sha256=original["reference_sha256"], source_sha256=sources,
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "status", "fixtures", "reference_executions", "score16_dispatches", "exercised")}), flush=True)


if __name__ == "__main__":
    main()
