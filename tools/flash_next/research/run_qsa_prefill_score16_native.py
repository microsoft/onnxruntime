import json
from pathlib import Path
import shlex
import subprocess

from build_qsa_prefill_score16 import FLAG
from prepare_qsa_prefill_score16_fixtures import DIRECTORY
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_qsa_indexer_stack_native import REPOS, UUID, programs


PROGRAM = "QsaPackedScore16QueryWide"


def verify_programs(before, after, admitted):
    if admitted:
        assert before[PROGRAM] == 0 and before["SparseAttentionIndexerQsaScore"] == 18
        assert after[PROGRAM] == 18 and after["SparseAttentionIndexerQsaScore"] == 0
    for name in set(before) | set(after):
        if not admitted or name not in (PROGRAM, "SparseAttentionIndexerQsaScore"):
            assert before[name] == after[name], name


def main():
    directory = REPOS / "qwen38-int4-qsa-prefill-score16/build-v1"
    prefix = ROOT / "qsa-prefill-score16-v1-native-gpu7-20261031"
    if any(prefix.with_suffix(s).exists() for s in (".qualification.json", ".guard.json")):
        raise FileExistsError(prefix)
    build = json.loads((directory / "build-manifest.json").read_text())
    parent = json.loads((ROOT / "dense-block32-v2-contexts-20261030.qualification.json").read_text())
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert build["baseline_byte_identical"] and build["baseline_provider_sha256"] == parent["provider_sha256"]
    provider = directory / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == build["provider_sha256"]
    frozen = json.loads((ROOT / "qsa-score16-v5-native-gpu7-20261029.qualification.json").read_text())
    extra = json.loads((DIRECTORY / "manifest.json").read_text())
    assert len(extra["fixtures"]) == 7 and extra["source_sha256"] == sha(
        Path(__file__).with_name("prepare_qsa_prefill_score16_fixtures.py"))
    fixtures = [Path(p) for p in sorted(frozen["fixture_sha256"])]
    assert len(fixtures) == 69
    digests = {**frozen["fixture_sha256"], **frozen["reference_sha256"]}
    admitted = set()
    for record in extra["fixtures"]:
        path = Path(record["path"])
        fixtures.append(path)
        digests[str(path)] = record["sha256"]
        digests[record["reference_path"]] = record["reference_sha256"]
        if record["candidate_admitted"]:
            admitted.add(path)
    assert len(fixtures) == 76 and len(admitted) == 4
    for path, digest in digests.items():
        assert sha(Path(path)) == digest
    comparator = REPOS / "onnxruntime-webgpu-h200-perf/build-int4-qsa-rank/qsa_rank_compare"
    assert sha(comparator) == frozen["comparator_sha256"]

    def mounted(path):
        return str(path).replace(str(REPOS), "/workspace").replace(str(ROOT), "/evidence/webgpu-perf-oct20")

    invocation = shlex.join([mounted(comparator), mounted(provider), *(mounted(p) for p in fixtures)])
    command = (
        "set -e; /workspace/onnxruntime-webgpu-h200-perf/build-int4-qsa-cache/vulkan_uuid_probe " + UUID +
        "; export ORT_WEBGPU_DYNAMIC_UNIFORMS=1 ORT_WEBGPU_UNIFORM_ARENA=1 "
        "ORT_WEBGPU_QSA_PARALLEL_RANK=1 ORT_WEBGPU_QSA_DENSE_MAP=1 "
        "ORT_WEBGPU_QSA_INDEXER_STACK=1 ORT_WEBGPU_QSA_SCORE16=1 "
        "ORT_WEBGPU_QSA_CONCAT_VECTOR=0 ORT_WEBGPU_QSA_PACKED_SCORE=0 "
        "ORT_TEST_QSA_RANK_COMPARE=1 ORT_TEST_QSA_CANDIDATE_ONLY=0; "
        f"for mode in 0 1; do export {FLAG}=$mode "
        f"ORT_TEST_PROFILE_PREFIX=/evidence/webgpu-perf-oct20/{prefix.name}-profile$mode; {invocation}; done")
    invalid = prefix.with_name(prefix.name + "-invalid.log")
    command += (
        f"; if {FLAG}=2 ORT_TEST_PROFILE_PREFIX={shlex.quote(mounted(prefix) + '-invalid-profile')} " +
        shlex.join([mounted(comparator), mounted(provider), mounted(fixtures[0])]) +
        f" > {shlex.quote(mounted(invalid))} 2>&1; then exit 1; fi; " +
        shlex.join(["grep", "-Fq", f"{FLAG} must be 0 or 1", mounted(invalid)]))
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
        "--output-prefix", str(prefix), "--", "bash", "-lc", command], check=True)
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    log = prefix.with_suffix(".log").read_text()
    assert guard["status"] == "completed" and guard["exit_code"] == 0
    assert guard["gpu_index"] == 7 and guard["gpu_uuid"] == UUID
    assert guard["command"] == ["bash", "-lc", command]
    assert "FAIL" not in log and "GetCapability" not in log
    assert log.count(" exact_checks=18 candidate_only=0 rank_compare=1 validation=full") == 152
    sources, coverage = {}, []
    for fixture in fixtures:
        events = []
        for mode in (0, 1):
            profiles = list(ROOT.glob(f"{prefix.name}-profile{mode}-{fixture.stem}_2026-*.json"))
            assert len(profiles) == 1, profiles
            events.append(json.loads(profiles[0].read_text()))
            sources[profiles[0].name] = sha(profiles[0])
        before, after = (programs(e) for e in events)
        verify_programs(before, after, fixture in admitted)
        if fixture in admitted:
            coverage.append(dict(fixture=fixture.name, dispatches=after[PROGRAM]))
    assert len(coverage) == 4 and len(sources) == 152
    assert f"{FLAG} must be 0 or 1" in invalid.read_text()
    negatives = list(ROOT.glob(prefix.name + "-invalid-profile-*.json"))
    assert len(negatives) == 1 and not programs(json.loads(negatives[0].read_text()))
    for path in (negatives[0], invalid, prefix.with_suffix(".guard.json"), prefix.with_suffix(".log"),
                 DIRECTORY / "manifest.json", Path(__file__)):
        sources[path.name] = sha(path)
    report = dict(status="prefill_score16_exact_reference_outputs_and_admission_qualified",
        provider_sha256=build["provider_sha256"], baseline_provider_sha256=build["baseline_provider_sha256"],
        fixtures=76, reference_executions=2736, exercised=coverage, selector=FLAG,
        invalid_selector_rejected=True, cpu_fallback_disabled=True, graph_optimizations_disabled=True,
        validation="full", original_shader_and_decode_admission_preserved=True,
        boundary_and_float32_fallbacks_qualified=True, two_dimensional_dispatch=True,
        distinct_prefill_cache_hint=True, build_manifest_sha256=sha(directory / "build-manifest.json"),
        comparator_sha256=sha(comparator), fixture_and_reference_sha256=digests, source_sha256=sources,
        instrumented_not_a_speed_claim=True, quality_qualified=False, headline_eligible=False)
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "reference_executions", "exercised")}), flush=True)


if __name__ == "__main__":
    main()
