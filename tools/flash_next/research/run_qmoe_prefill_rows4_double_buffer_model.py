import argparse
import collections
import json
from pathlib import Path
import subprocess
import time

from build_qmoe_prefill_rows4_double_buffer import FLAG, NAME
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from prepare_qmoe_prefill_rows4_double_buffer_model import BENCHMARK
from qualify_quickgelu_mul_long_context import sha
from run_prefill256_model import verify_metadata, verify_prefill_flags, verify_tokens
from run_qmoe_prefill_rows4_double_buffer_native import BUILD, ORIGINAL, ROOT
from run_qmoe_prefill_rows4_standard import TEMPLATE, standard_command


VARIANT = "qmoe-prefill-rows4-double-buffer-v2"


def prefix_for(context, full, phase, mode, trial):
    return ROOT / f"{VARIANT}-mode{mode}-gpu7-{'full-' if full else ''}{phase}{context}-trial{trial}"


def model_command(template, provider, prefix, context, full, phase, mode):
    command = standard_command(template, provider, prefix, context, full)
    original_benchmark = "/evidence/webgpu-perf-oct20/benchmark_webgpu_int4.py"
    assert command[-1].count(original_benchmark) == 1
    command[-1] = command[-1].replace(original_benchmark,
        f"/evidence/webgpu-perf-oct20/{BENCHMARK.name}")
    marker = "--tokens 256 --repetitions 3"
    assert command[-1].count(marker) == 1
    tokens = 2 if phase == "profile" else 36
    command[-1] = command[-1].replace(marker, f"--tokens {tokens} --repetitions 1")
    command[-1] = f"export {FLAG}={mode}; " + command[-1]
    if phase == "profile":
        command[-1] += f" --profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile"
    return command


def verify_profile(counts, context):
    expected = 96 * (context // 1024)
    assert counts[NAME] == expected, counts
    assert counts["MatMulNBitsGroupedRows4F16"] == expected, counts
    assert counts["HCPostNormProjectionWideRmsBroadcastF16"] == 0, counts
    assert counts["QmoeFC1SwiGLUDoubleBufferF16"] == 0, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--phase", choices=("profile", "screen"), default="profile")
    parser.add_argument("--mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--trial", type=int, choices=range(4), default=0)
    parser.add_argument("--verify-existing-profile", action="store_true")
    args = parser.parse_args()
    if args.phase == "profile" and (not args.full_validation or args.mode != 1 or args.trial != 0):
        parser.error("Profile requires full validation, candidate selector1, trial0")
    if args.verify_existing_profile and args.phase != "profile":
        parser.error("Existing evidence verification is restricted to profiling")
    prefix = prefix_for(args.context, args.full_validation, args.phase, args.mode, args.trial)
    for suffix in (".qualification.json", ".json", ".log", ".guard.json"):
        path = prefix.with_suffix(suffix)
        if path.exists() and (suffix == ".qualification.json" or not args.verify_existing_profile):
            raise FileExistsError(prefix.with_suffix(suffix))
        if args.verify_existing_profile and suffix != ".qualification.json" and not path.is_file():
            raise FileNotFoundError(path)
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    parent = json.loads(parent_path.read_text())
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    evidence = [Path(__file__), Path(__file__).with_name("build_qmoe_prefill_rows4_double_buffer.py"),
                parent_path, BENCHMARK,
                Path(__file__).with_name("prepare_qmoe_prefill_rows4_double_buffer_model.py")]
    for label, expected in (("native", "prefill_separate_original_outputs_supported"),
                            ("checks", "fresh_selector_and_original_replay_fallback_supported")):
        path = ROOT / f"{VARIANT}-{label}-outcome.json"
        proof = json.loads(path.read_text())
        assert proof["status"] == expected
        evidence.append(path)
        for name, digest in proof["source_and_evidence_sha256"].items():
            assert sha(Path(name)) == digest, name
            evidence.append(Path(name))
    build_path = BUILD / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert sha(evidence[1]) == build["builder_sha256"]
    assert build["baseline_provider_sha256"] == parent["provider_sha256"]
    assert build["baseline_byte_identical"] and build["all_152_original_source_objects_byte_identical"]
    for name, digest in build["source_sha256"].items():
        assert sha(BUILD / name) == digest, name
        evidence.append(BUILD / name)
    provider = BUILD / "libonnxruntime_providers_webgpu.so" if args.mode else ORIGINAL
    provider_hash = build["provider_sha256"] if args.mode else parent["provider_sha256"]
    assert sha(provider) == provider_hash
    for name, digest in parent["source_sha256"].items():
        assert sha(Path(name)) == digest, name
    evidence.extend([provider, build_path])
    if args.phase == "screen":
        path = prefix_for(args.context, True, "profile", 1, 0).with_suffix(".qualification.json")
        proof = json.loads(path.read_text())
        assert proof["status"] == "original_sequences_qualified"
        assert proof["provider_sha256"] == build["provider_sha256"]
        for name, digest in proof["source_sha256"].items():
            assert sha(Path(name)) == digest, name
        evidence.append(path)
    if args.context != 2048:
        previous = 2048 if args.context == 8192 else 8192
        path = ROOT / f"{VARIANT}-context{previous}-ttft-screen.qualification.json"
        proof = json.loads(path.read_text())
        assert proof["status"] == "paired_ttft_win_both_modes"
        assert proof["provider_sha256"] == build["provider_sha256"]
        for name, digest in proof["source_sha256"].items():
            assert sha(Path(name)) == digest, name
        evidence.append(path)
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    template_raw_path = ROOT / f"{TEMPLATE}.json"
    template = json.loads(template_path.read_text())
    command = model_command(template["command"], provider, prefix, args.context,
                            args.full_validation, args.phase, args.mode)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    if args.verify_existing_profile:
        cache = json.loads((ROOT / (prefix.name + "-cache.json")).read_text())
        assert "after_run" in cache
        started_ns = completed_ns = None
    else:
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
    verify_metadata(raw, guard, {"provider_sha256": provider_hash}, parent, args.full_validation, chunk=1024)
    verify_prefill_flags(raw, 1024)
    assert raw["load_experiment_flags"][FLAG] == str(args.mode)
    assert raw["load_experiment_flags"]["ORT_WEBGPU_QMOE_GROUPED_PREFILL"] == "1"
    assert raw["cpu_affinity"] == list(range(96)) and "worker_affinity_override" not in raw
    assert raw["genai_native"] == json.loads(template_raw_path.read_text())["genai_native"]
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [row for row in json.loads(controls_path.read_text())["samples"] if row["context"] == args.context]
    tokens = 2 if args.phase == "profile" else 36
    verify_tokens(raw, references, args.context, tokens)
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    evidence.extend([template_path, template_raw_path, controls_path, Path(cache["evidence_path"])])
    for suffix in (".json", ".guard.json", ".log"):
        evidence.append(prefix.with_suffix(suffix))
    counts = collections.Counter()
    if args.phase == "profile":
        profiles = list(ROOT.glob(prefix.name + "-profile*.json"))
        assert raw["profiling_scope"] == "session" and len(profiles) == 1, profiles
        for path in profiles:
            counts.update(event["name"].rsplit("&", 1)[-1] for event in json.loads(path.read_text())
                          if event.get("cat") == "Kernel")
        verify_profile(counts, args.context)
        evidence.extend(profiles)
    report = dict(status="original_sequences_qualified", context=args.context,
        full_validation=args.full_validation, phase=args.phase, mode=args.mode, trial=args.trial,
        provider_sha256=provider_hash, candidate_provider_sha256=build["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        original_output_prefixes_match=True, zero_major_faults=True,
        program_counts=dict(counts), started_ns=started_ns, completed_ns=completed_ns,
        verified_existing_profile=args.verify_existing_profile,
        short_screen_not_standard_decode=True,
        source_sha256={str(path): sha(path) for path in evidence})
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], context=args.context, phase=args.phase,
                         mode=args.mode, full=args.full_validation, program_counts=dict(counts))), flush=True)


if __name__ == "__main__":
    main()
