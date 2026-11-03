import argparse
import collections
import json
from pathlib import Path
import subprocess
import time

from build_dense_packed_columns import FLAG
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_tokens
from run_qmoe_prefill_rows4_standard import TEMPLATE, standard_command, verify_result


VARIANT = "dense-packed-columns-v2"
STAMP = "20261031"
DIRECTORY = Path("/home/sayanshaw/repos/qwen38-int4-dense-packed-columns/build-v2")


def prefix_for(mode, context, full, trial, phase="standard"):
    return ROOT / f"{VARIANT}-{mode}-gpu7-{'full-' if full else ''}{phase}{context}-trial{trial}-{STAMP}"


def model_command(template, provider, prefix, context, full, mode, phase):
    assert mode in ("control", "packed_columns") and phase in ("profile", "standard")
    command = standard_command(template, provider, prefix, context, full)
    command[-1] = f"export {FLAG}={int(mode == 'packed_columns')}; " + command[-1]
    if phase == "profile":
        assert full
        marker = "--tokens 256 --repetitions 3"
        assert command[-1].count(marker) == 1
        command[-1] = command[-1].replace(
            marker, "--tokens 36 --repetitions 1 --profile-decode-after 35 "
            f"--profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile")
    return command


def verify_profile(counts, mode):
    selected = "MatMulNBitsPackedColumnsF16" if mode == "packed_columns" else "MatMulNBitsBlock32F16"
    original = "MatMulNBitsBlock32F16" if mode == "packed_columns" else "MatMulNBitsPackedColumnsF16"
    assert counts[selected] == 159 and counts[original] == 0, counts
    assert counts["MatMulNBitsTile4F16"] == 0, counts
    assert counts["DynamicSparseAttentionQueryCache"] == 12, counts
    assert counts["DynamicSparseAttention"] == 0, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("control", "packed_columns"), default="packed_columns")
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--phase", choices=("profile", "standard"), default="standard")
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--trial", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.phase == "profile" and not args.full_validation:
        parser.error("Active-path profiling requires full validation")
    if args.mode == "packed_columns" and args.trial == 2:
        parser.error("Trial2 is reserved for the final control")
    prefix = prefix_for(args.mode, args.context, args.full_validation, args.trial, args.phase)
    for suffix in (".json", ".guard.json", ".log", ".qualification.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    native_path = ROOT / f"{VARIANT}-native-gpu7-{STAMP}.qualification.json"
    fp32_path = ROOT / f"{VARIANT}-fp32-native-gpu7-{STAMP}.qualification.json"
    parent, native, fp32 = (json.loads(path.read_text()) for path in
                            (parent_path, native_path, fp32_path))
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert native["status"] == "dense_packed_columns_changing_inputs_finite_bitwise_and_replay_qualified"
    assert fp32["status"] == "dense_packed_columns_fp32_accumulation_fallback_finite_bitwise_qualified"
    assert native["finite_bitwise_comparisons"] == fp32["finite_bitwise_comparisons"] == 116582784
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == native["provider_sha256"] == fp32["provider_sha256"]
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    build_path = DIRECTORY / "build-manifest.json"
    assert sha(build_path) == native["build_manifest_sha256"] == fp32["build_manifest_sha256"]
    build = json.loads(build_path.read_text())
    for name, digest in build["source_sha256"].items():
        assert sha(DIRECTORY / name) == digest, name
    for name, digest in parent["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    candidate_native_path = native_path
    candidate_native = native
    if args.mode == "control":
        native_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
        native = json.loads(native_path.read_text())
        provider = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-prefill-rows4/build-v2/"
                        "libonnxruntime_providers_webgpu.so")
        assert sha(provider) == native["provider_sha256"] == parent["provider_sha256"]
    prerequisites = []
    if args.phase == "standard":
        prerequisites.append(prefix_for("packed_columns", 2048, True, 0, "profile").with_suffix(".qualification.json"))
    if args.phase == "standard" and not args.full_validation:
        prerequisites.append(prefix_for(args.mode, args.context, True, args.trial).with_suffix(".qualification.json"))
    if args.mode == "packed_columns" and args.trial == 1:
        prerequisites.append(ROOT / f"{VARIANT}-context{args.context}-screen-{STAMP}.qualification.json")
    if args.context != 2048:
        previous = 2048 if args.context == 8192 else 8192
        prerequisites.append(ROOT / f"{VARIANT}-context{previous}-repeat-screen-{STAMP}.qualification.json")
    for path in prerequisites:
        proof = json.loads(path.read_text())
        expected = ("repeated_packed_columns_decode_gain_both_modes" if "repeat-screen" in path.name else
                    "initial_packed_columns_decode_gain_both_modes" if "-screen-" in path.name else
                    "original_sequences_qualified")
        gate_provider = (candidate_native["provider_sha256"] if "-screen-" in path.name or
                         "-profile" in path.name else native["provider_sha256"])
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
    started_ns = time.time_ns()
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
                    "--output-prefix", str(prefix), "--", *command], check=True)
    completed_ns = time.time_ns()
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = (json.loads(prefix.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json"))
    assert guard["command"] == command
    assert raw["cpu_affinity"] == list(range(96)) and "worker_affinity_override" not in raw
    assert any(len(mask) == 1 for mask in raw["thread_cpu_affinity"].values())
    assert raw["genai_native"]["sha256"] == sha(Path(
        "/home/sayanshaw/repos/onnxruntime-genai-webgpu-fp8-checkpoint/build-h200-int4-legacy/libonnxruntime-genai.so"))
    assert raw["comparison"] is None
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [sample for sample in json.loads(controls_path.read_text())["samples"]
                  if sample["context"] == args.context]
    profiles = []
    if args.phase == "standard":
        assert not raw["profiling"] and raw["profiling_scope"] is None
        result = verify_result(raw, guard, native, parent, references, args.context, args.full_validation)
    else:
        from run_prefill256_model import verify_metadata
        verify_metadata(raw, guard, native, parent, True, chunk=1024)
        verify_tokens(raw, references, args.context, 36)
        assert raw["profiling_scope"] == "decode_only" and raw["profile_decode_after_tokens"] == 35
        profiles = list(ROOT.glob(prefix.name + f"-profile-ctx{args.context}-rep0-configured_*.json"))
        assert len(profiles) == 1, profiles
        counts = collections.Counter(event["name"].rsplit("&", 1)[-1]
                                     for event in json.loads(profiles[0].read_text()) if event.get("cat") == "Kernel")
        verify_profile(counts, args.mode)
        result = dict(program_counts=dict(counts), instrumented_not_a_speed_claim=True)
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    sources = [Path(__file__), ROOT / "benchmark_webgpu_int4.py", parent_path, native_path, fp32_path,
               build_path, candidate_native_path, template_path, controls_path, *prerequisites, *profiles,
               *[prefix.with_suffix(suffix) for suffix in (".json", ".guard.json", ".log")]]
    report = dict(status="original_sequences_qualified", context=args.context, mode=args.mode,
                  phase=args.phase, trial=args.trial, full_validation=args.full_validation,
                  provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
                  graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
                  original_provider_control=args.mode == "control", experiment_flag=FLAG, experiment_flag_value=int(args.mode == "packed_columns"),
                  original_output_prefixes_match=True, zero_major_faults=True, result=result,
                  worker_policy_unchanged=True, started_ns=started_ns, completed_ns=completed_ns,
                  cache_preflight=cache, quality_qualified=False, headline_eligible=False,
                  source_sha256={str(path): sha(path) for path in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("status", "mode", "context", "phase", "full_validation", "result")}), flush=True)


if __name__ == "__main__":
    main()
