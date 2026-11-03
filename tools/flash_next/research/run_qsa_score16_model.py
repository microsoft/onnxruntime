import argparse
import collections
import json
import math
from pathlib import Path
import statistics
import subprocess

from build_qsa_score16 import FLAG
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_tokens
from run_prefill_dense_model import verify_windows


def verify_paired(raw, references, context, *, prefix="qsa_score16", experiment_flag=FLAG):
    assert raw["comparison"] == f"{prefix}_decode_windows"
    assert raw["window_protocol"]["experiment_flag"] == experiment_flag
    assert len(raw["samples"]) == 4
    reference = references[1]
    for sample in raw["samples"]:
        assert sample["context"] == context
        assert sample["prompt_sha256"] == reference["prompt_sha256"]
        assert len(sample["output_token_ids"]) == sample["output_tokens"] == 288
        assert sample["output_token_ids"][:256] == reference["output_token_ids"]
        assert sample["output_token_ids"] == raw["samples"][0]["output_token_ids"]
        assert sample["same_prompt_output_matches"]
        assert all(v["major_page_faults"] == 0 for v in sample["process_resource_usage"].values())
    result = verify_windows(raw, context, prefix=prefix)
    for row in raw["window_samples"]:
        assert all(math.isfinite(row[f]) and row[f] > 0 for f in ("decode_tps", "decode_seconds"))
    groups = [(raw["window_samples"], result)]
    assert len(result["per_request"]) == 4
    assert {r["repetition"] for r in result["per_request"]} == set(range(4))
    for request in result["per_request"]:
        groups.append(([r for r in raw["window_samples"] if r["repetition"] == request["repetition"]], request))
    for rows, summary in groups:
        for mode in (0, 1):
            expected = statistics.median(r["decode_tps"] for r in rows if r[f"{prefix}_flag"] == mode)
            assert math.isclose(summary[f"{prefix}{mode}_median_window_tps"], expected, rel_tol=1e-12)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=32768)
    parser.add_argument("--phase", choices=("profile", "windows", "standard"), default="windows")
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--version", type=int, choices=(2, 5), default=2)
    args = parser.parse_args()
    active_contexts = (32768,) if args.version == 2 else (8192, 32768)
    if args.phase == "profile" and (not args.full_validation or args.context not in active_contexts):
        parser.error("Active-path profiling requires full validation at an eligible context")
    variant = f"qsa-score16-v{args.version}"
    program_name = "QsaPackedScore16" if args.version == 2 else "QsaPackedScore16QueryWide"
    directory = Path(f"/home/sayanshaw/repos/qwen38-int4-qsa-score16/build-v{args.version}")
    provider = directory / "libonnxruntime_providers_webgpu.so"
    native = json.loads((ROOT / f"{variant}-native-gpu7-20261029.qualification.json").read_text())
    parent = json.loads((ROOT / "prefill256-v1-contexts-20261029.qualification.json").read_text())
    assert native["status"] == "exact_reference_and_score16_dispatch_qualified"
    assert native["minimum_score_stride"] == (8192 if args.version == 2 else 2048)
    assert native["reference_executions"] == 2484
    if args.version == 5:
        assert native["program_name"] == program_name
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert sha(provider) == native["provider_sha256"]
    if not args.full_validation:
        validation_context = 8192 if args.version == 5 and args.context != 32768 else 32768
        validation = json.loads((ROOT / (
            f"{variant}-chunk256-gpu7-full-windows{validation_context}-20261029.qualification.json")).read_text())
        assert validation["provider_sha256"] == native["provider_sha256"]
        assert validation["full_validation"] and validation["zero_major_faults"]
        assert validation["original_output_prefixes_match"]
    prefix = ROOT / (f"{variant}-chunk256-gpu7-{'full-' if args.full_validation else ''}"
                     f"{args.phase}{args.context}-20261029")
    if prefix.with_suffix(".qualification.json").exists() or (
            not args.verify_existing and any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json"))):
        raise FileExistsError(prefix)
    baseline = "prefill256-v1-gpu7-full-standard2048-20261029"
    command = json.loads((ROOT / (baseline + ".guard.json")).read_text())["command"]
    old_provider = "/workspace/qwen38-int4-qmoe-prefill256/build-v1/libonnxruntime_providers_webgpu.so"
    assert command[-1].count(old_provider) == command[-1].count("--chunk 256") == 1
    command[-1] = (f"export {FLAG}=1; " + command[-1].replace(
        old_provider, str(provider).replace("/home/sayanshaw/repos", "/workspace")).replace(
        baseline, prefix.name).replace("--contexts 2048", f"--contexts {args.context}"))
    marker = "--tokens 256 --repetitions 3"
    assert command[-1].count(marker) == 1
    if args.phase == "windows":
        replacement = "--tokens 288 --repetitions 4 --compare-qsa-score16-windows"
    elif args.phase == "profile":
        replacement = ("--tokens 36 --repetitions 1 --profile-decode-after 35 "
                       f"--profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile")
    else:
        replacement = marker
    command[-1] = command[-1].replace(marker, replacement)
    if not args.full_validation:
        assert command[-1].count(" --webgpu-validation-mode full") == 1
        command[-1] = command[-1].replace(" --webgpu-validation-mode full", "")
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    if args.verify_existing:
        cache = json.loads(prefix.with_name(prefix.name + "-cache.json").read_text())
        assert "after_run" in cache
    else:
        cache = prepare_model_cache(model, prefix)
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
            "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
            "--output-prefix", str(prefix), "--", *command], check=True)
        cache["after_run"] = snapshot_model_cache(model)
        Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw = json.loads(prefix.with_suffix(".json").read_text())
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    verify_metadata(raw, guard, native, parent, args.full_validation)
    assert guard["command"] == command and raw["load_experiment_flags"][FLAG] == "1"
    assert raw["graph_sha256"] == sha(model / "model.onnx")
    assert raw["config_sha256"] == sha(model / "genai_config.json")
    controls = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())
    references = [s for s in controls["samples"] if s["context"] == args.context]
    profiles = []
    if args.phase == "windows":
        result = verify_paired(raw, references, args.context)
    else:
        result = verify_tokens(raw, references, args.context, 36 if args.phase == "profile" else 256)
        if args.phase == "profile":
            assert raw["profiling_scope"] == "decode_only" and raw["profile_decode_after_tokens"] == 35
            profiles = list(ROOT.glob(prefix.name + f"-profile-ctx{args.context}-rep0-configured_*.json"))
            assert len(profiles) == 1, profiles
            counts = collections.Counter(e["name"].rsplit("&", 1)[-1] for e in json.loads(profiles[0].read_text())
                                         if e.get("cat") == "Kernel")
            assert counts[program_name] == 12 and counts["QsaPackedScore4"] == 0, counts
            result = dict(candidate_program_dispatches={program_name: 12},
                          instrumented_not_a_speed_claim=True)
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    report = dict(status="original_sequences_qualified", context=args.context, phase=args.phase,
        full_validation=args.full_validation, chunk=256, zero_major_faults=True,
        original_output_prefixes_match=True, provider_sha256=native["provider_sha256"],
        baseline_provider_sha256=parent["provider_sha256"], graph_sha256=parent["graph_sha256"],
        config_sha256=parent["config_sha256"], result=result, cache_preflight=cache,
        source_sha256={p.name: sha(p) for p in [*profiles, prefix.with_suffix(".json"),
                                               prefix.with_suffix(".guard.json")]},
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "context", "phase", "result")}), flush=True)


if __name__ == "__main__":
    main()
