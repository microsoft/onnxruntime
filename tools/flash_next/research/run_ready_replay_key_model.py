import argparse
import json
from pathlib import Path
import subprocess

from build_ready_replay_key_trim import FLAG
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_tokens
from run_qmoe_prefill_rows4_standard import TEMPLATE, standard_command, verify_result


VARIANT = "ready-replay-key-trim-v2"
DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-ready-replay-key-trim/build-v2")


def prefix_for(mode, context, full, trial, phase="standard"):
    return ROOT / f"{VARIANT}-{mode}-gpu7-{'full-' if full else ''}{phase}{context}-trial{trial}-20261031"


def model_command(template, provider, prefix, context, full, mode, phase):
    assert mode in ("control", "trim") and phase in ("coverage", "standard")
    command = standard_command(template, provider, prefix, context, full)
    command[-1] = (
        f"export {FLAG}={int(mode == 'trim')}; "
        f"export ORT_WEBGPU_READY_REPLAY_KEY_STATS_FILE=/evidence/webgpu-perf-oct20/"
        f"{prefix.name}-key-stats.jsonl; " + command[-1])
    if phase == "coverage":
        assert full
        marker = "--tokens 256 --repetitions 3"
        assert command[-1].count(marker) == 1
        command[-1] = command[-1].replace(marker, "--tokens 36 --repetitions 1")
    return command


def verify_coverage(rows):
    dense = [row for row in rows if row.get("operator") == "MatMulNBitsExact"]
    assert len(dense) == 171, len(dense)
    assert all(row["hits"] > 0 and row["binding_hits"] > 0 and
               row["rebinds"] + row["binding_hits"] == row["hits"] for row in dense)
    return dict(actual_dense_replay_instances=len(dense),
                dense_replay_hits=sum(row["hits"] for row in dense),
                dense_binding_hits=sum(row["binding_hits"] for row in dense),
                actual_replay_path_coverage_only=True, not_a_standard_tps_claim=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("control", "trim"), default="trim")
    parser.add_argument("--phase", choices=("coverage", "standard"), default="standard")
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--trial", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.phase == "coverage" and (not args.full_validation or args.context != 2048):
        parser.error("Coverage requires full 2K validation")
    if args.mode == "trim" and args.trial == 2:
        parser.error("Trial2 is the final control")
    prefix = prefix_for(args.mode, args.context, args.full_validation, args.trial, args.phase)
    for suffix in (".json", ".guard.json", ".log", ".qualification.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    native_path = ROOT / f"{VARIANT}-native-gpu7-20261031.qualification.json"
    parent, native = (json.loads(p.read_text()) for p in (parent_path, native_path))
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert native["status"] == "ready_replay_key_trim_process_modes_and_invalid_selector_qualified"
    assert native["finite_bitwise_comparisons"] == 233165568
    assert native["process_modes_tested"] == [0, 1]
    assert native["invalid_selector_rejected_in_fresh_process"]
    for name, digest in native["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    build_path = DIRECTORY / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert sha(build_path) == native["build_manifest_sha256"]
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == native["provider_sha256"] == build["provider_sha256"]
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert sha(DIRECTORY / "native_compare") == native["comparator_sha256"]
    for name, digest in build["source_sha256"].items():
        assert sha(DIRECTORY / name) == digest, name
    candidate_native = native
    candidate_native_path = native_path
    original_native_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
    original_native = json.loads(original_native_path.read_text())
    assert original_native["provider_sha256"] == parent["provider_sha256"]
    if args.mode == "control":
        native = original_native
        native_path = original_native_path
        provider = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/"
                        "libonnxruntime_providers_webgpu.so")
        assert sha(provider) == native["provider_sha256"]
    prerequisites = []
    if args.phase == "standard":
        prerequisites.append(prefix_for("trim", 2048, True, 0, "coverage").with_suffix(".qualification.json"))
        if not args.full_validation:
            prerequisites.append(prefix_for(args.mode, args.context, True, args.trial).with_suffix(".qualification.json"))
        if args.mode == "trim" and args.trial == 1:
            prerequisites.append(ROOT / f"{VARIANT}-context{args.context}-screen-20261031.qualification.json")
        if args.context != 2048:
            previous = 2048 if args.context == 8192 else 8192
            prerequisites.append(ROOT / f"{VARIANT}-context{previous}-repeat-screen-20261031.qualification.json")
    for path in prerequisites:
        proof = json.loads(path.read_text())
        expected = ("repeated_trim_decode_gain_both_modes" if "repeat-screen" in path.name else
                    "initial_trim_decode_gain_both_modes" if "-screen-" in path.name else
                    "original_sequences_qualified")
        gate_provider = (candidate_native["provider_sha256"] if "-screen-" in path.name or
                         "coverage" in path.name else native["provider_sha256"])
        assert proof["status"] == expected and proof["provider_sha256"] == gate_provider
        for name, digest in proof["source_sha256"].items():
            assert sha(Path(name)) == digest, name
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    command = model_command(json.loads(template_path.read_text())["command"], provider, prefix,
                            args.context, args.full_validation, args.mode, args.phase)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
                    "--output-prefix", str(prefix), "--", *command], check=True)
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = (json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json"))
    assert guard["command"] == command and raw["comparison"] is None
    assert not raw["profiling"] and raw["profiling_scope"] is None
    assert raw["cpu_affinity"] == list(range(96)) and "worker_affinity_override" not in raw
    assert any(len(mask) == 1 for mask in raw["thread_cpu_affinity"].values())
    assert raw["genai_native"]["sha256"] == sha(Path(
        "/home/sayanshaw/repos/onnxruntime-genai-webgpu-fp8-checkpoint/build-h200-int4-legacy/libonnxruntime-genai.so"))
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [s for s in json.loads(controls_path.read_text())["samples"] if s["context"] == args.context]
    statistics = []
    if args.phase == "standard":
        result = verify_result(raw, guard, native, parent, references, args.context, args.full_validation)
    else:
        verify_metadata(raw, guard, native, parent, True, chunk=1024)
        verify_tokens(raw, references, args.context, 36)
        stats_path = ROOT / (prefix.name + "-static-stats.jsonl")
        result = verify_coverage([json.loads(s) for s in stats_path.read_text().splitlines()])
        statistics.append(stats_path)
    key_path = ROOT / (prefix.name + "-key-stats.jsonl")
    key_coverage = None
    if args.mode == "trim":
        rows = [json.loads(line) for line in key_path.read_text().splitlines()]
        assert rows and all(row["operator"] == "ReadyReplayWithoutKey" and
                            row["commands"] > 0 and row["key_bytes_avoided"] > 0
                            for row in rows), rows
        key_coverage = dict(commands=sum(row["commands"] for row in rows),
                            key_bytes_avoided=sum(row["key_bytes_avoided"] for row in rows))
        if args.phase == "coverage":
            assert key_coverage["commands"] >= result["dense_replay_hits"]
        statistics.append(key_path)
    else:
        assert not key_path.exists()
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    sources = [Path(__file__), ROOT / "benchmark_webgpu_int4.py", parent_path, native_path,
               build_path, candidate_native_path, original_native_path,
               template_path, controls_path, *prerequisites, *statistics,
               *[prefix.with_suffix(s) for s in (".json", ".guard.json", ".log")]]
    report = dict(status="original_sequences_qualified", context=args.context, mode=args.mode,
                  phase=args.phase, trial=args.trial, full_validation=args.full_validation,
                  provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
                  original_provider_control=args.mode == "control", process_static_experiment=True,
                  ready_replay_key_coverage=key_coverage,
                  graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
                  experiment_flag=FLAG, experiment_flag_value=int(args.mode == "trim"),
                  original_output_prefixes_match=True, zero_major_faults=True, result=result,
                  worker_policy_unchanged=True, cache_preflight=cache,
                  quality_qualified=False, headline_eligible=False,
                  source_sha256={str(p): sha(p) for p in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "mode", "phase", "context", "result")}), flush=True)


if __name__ == "__main__":
    main()
