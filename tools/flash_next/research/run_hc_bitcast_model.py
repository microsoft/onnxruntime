import argparse
import collections
import json
from pathlib import Path
import subprocess

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill_dense_model import verify_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--phase", choices=("windows", "standard", "profile"), default="windows")
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--candidate", choices=("bitcast", "fused-sum", "subgroup", "up-bitcast",
                                               "concat-vector", "packed-score", "indexer-stack"),
                        default="bitcast")
    parser.add_argument("--version", type=int, choices=(1, 2, 3))
    parser.add_argument("--chunk", type=int, choices=(64, 128), default=64)
    args = parser.parse_args()
    if args.phase == "profile" and not args.full_validation:
        parser.error("The diagnostic profiles one fully validated decode step")
    version = args.version or {"bitcast": 1, "fused-sum": 3, "subgroup": 2, "up-bitcast": 1,
                               "concat-vector": 1, "packed-score": 2, "indexer-stack": 1}[args.candidate]
    if args.candidate == "bitcast":
        variant = f"hc-bitcast-unpack-v{version}"
        flag = "ORT_WEBGPU_HC_POST_NORM_BITCAST"
        program_name = "HCPostNormProjectionBitcastF16"
    elif args.candidate == "fused-sum":
        variant = f"hc-fused-sum-v{version}"
        flag = "ORT_WEBGPU_HC_POST_NORM_FUSED_SUM"
        program_name = "HCPostNormProjectionFusedSumF16"
    elif args.candidate == "subgroup":
        variant = f"hc-subgroup-norm-v{version}"
        flag = "ORT_WEBGPU_HC_POST_NORM_SUBGROUP"
        program_name = "HCPostNormProjectionSubgroupNormF16"
    elif args.candidate == "indexer-stack":
        variant = f"qsa-indexer-stack-v{version}"
        flag = "ORT_WEBGPU_QSA_INDEXER_STACK"
        program_name = "QsaPackedScore4"
    elif args.candidate == "packed-score":
        variant = f"qsa-packed-score-v{version}"
        flag = "ORT_WEBGPU_QSA_PACKED_SCORE"
        program_name = "QsaPackedScore4"
    elif args.candidate == "concat-vector":
        variant = f"qsa-concat-vector-v{version}"
        flag = "ORT_WEBGPU_QSA_CONCAT_VECTOR"
        program_name = "QsaConcatVector4"
    else:
        variant = f"hc-up-bitcast-unpack-v{version}"
        flag = "ORT_WEBGPU_HC_UP_BITCAST_UNPACK"
        program_name = "HCUpPreMixBitcastF16"
    evidence_tag = variant + ("-chunk128" if args.chunk == 128 else "")
    mode_prefix = flag.removeprefix("ORT_WEBGPU_").lower()
    directory = Path(f"/home/sayanshaw/repos/qwen38-int4-{variant.removesuffix(f'-v{version}')}/build-v{version}")
    provider = directory / "libonnxruntime_providers_webgpu.so"
    proof = json.loads((ROOT / f"{variant}-native-gpu7-20261029.qualification.json").read_text())
    parent = json.loads((ROOT / "prefill-dense-stack-v1-native-gpu7-20261029.qualification.json").read_text())
    model_proof = json.loads((ROOT / "qmoe-extended-prefill-v1-contexts-20261029.qualification.json").read_text())
    expected_status = ("exact_reference_and_indexer_stack_dispatch_qualified" if args.candidate == "indexer-stack" else
                       "exact_reference_and_packed_score_dispatch_qualified" if args.candidate == "packed-score" else
                       "exact_reference_and_vector_concat_dispatch_qualified" if args.candidate == "concat-vector" else
                       "hc_up_bitcast_finite_bitwise_and_replay_qualified" if args.candidate == "up-bitcast"
                       else f"hc_{args.candidate.replace('-', '_')}_four_outputs_finite_bitwise_and_replay_qualified")
    assert proof["status"] == expected_status
    assert sha(provider) == proof["provider_sha256"]
    assert proof["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["baseline_provider_sha256"] == model_proof["provider_sha256"]
    if args.chunk == 128:
        prefill_proof = json.loads((ROOT / "prefill128-dense-stack-v1-contexts-20261029.qualification.json").read_text())
        assert prefill_proof["provider_sha256"] == proof["baseline_provider_sha256"]
        assert prefill_proof["all_original_sequences_match"] and prefill_proof["zero_major_faults"]
    if not args.full_validation:
        validation_context = 8192 if (args.candidate == "indexer-stack" or
                                     args.candidate == "packed-score" and version >= 2) else 2048
        validation = json.loads((ROOT / (
            f"{evidence_tag}-gpu7-full-windows{validation_context}-20261029.qualification.json")).read_text())
        assert validation["provider_sha256"] == proof["provider_sha256"]
        assert validation["original_output_prefixes_match"] and validation["zero_major_faults"]
    baseline = "dense-bitcast-unpack-v1-gpu7-windows-20261029"
    command = json.loads((ROOT / (baseline + ".guard.json")).read_text())["command"]
    prefix = ROOT / (f"{evidence_tag}-gpu7-{'full-' if args.full_validation else ''}"
                     f"{args.phase}{args.context}-20261029")
    if prefix.with_suffix(".qualification.json").exists() or (
            not args.verify_existing and any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json"))):
        raise FileExistsError(prefix)
    original_provider = "/workspace/qwen38-int4-dense-bitcast-unpack/build-v1/libonnxruntime_providers_webgpu.so"
    assert command[-1].count(original_provider) == 1
    command[-1] = command[-1].replace(original_provider, str(provider).replace(
        "/home/sayanshaw/repos", "/workspace")).replace(baseline, prefix.name).replace(
        "--contexts 2048", f"--contexts {args.context}")
    marker = "--tokens 288 --repetitions 4 --compare-dense-bitcast-unpack-windows"
    assert marker in command[-1]
    if args.phase == "windows":
        comparison = ("--compare-qsa-indexer-stack-windows" if args.candidate == "indexer-stack" else
                      "--compare-qsa-packed-score-windows" if args.candidate == "packed-score" else
                      "--compare-qsa-concat-vector-windows" if args.candidate == "concat-vector" else
                      "--compare-hc-up-bitcast-unpack-windows" if args.candidate == "up-bitcast"
                      else f"--compare-hc-post-norm-{args.candidate}-windows")
        replacement = f"--tokens 288 --repetitions 4 {comparison}"
    elif args.phase == "standard":
        replacement = "--tokens 256 --repetitions 3"
    else:
        replacement = ("--tokens 36 --repetitions 1 --profile-decode-after 35 "
                       f"--profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile")
    child_flags = (" ORT_WEBGPU_QSA_CONCAT_VECTOR=0 ORT_WEBGPU_QSA_PACKED_SCORE=0"
                   if args.candidate == "indexer-stack" else "")
    command[-1] = (f"export ORT_WEBGPU_QMOE_EXTENDED_PREFILL=1 {flag}=1{child_flags}; " +
                    command[-1].replace(marker, replacement) + f" --chunk {args.chunk}" +
                    (" --webgpu-validation-mode full" if args.full_validation else ""))
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    if args.verify_existing:
        cache = json.loads(prefix.with_name(prefix.name + "-cache.json").read_text())
        assert "after_run" in cache
    else:
        cache = prepare_model_cache(model, prefix)
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
            "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "1800",
            "--output-prefix", str(prefix), "--", *command], check=True)
        cache["after_run"] = snapshot_model_cache(model)
        Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    raw = json.loads(prefix.with_suffix(".json").read_text())
    assert guard["status"] == raw["status"] == "completed" and guard["exit_code"] == 0
    assert guard["command"] == command and guard["gpu_index"] == 7
    assert guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
    assert raw["plugin_sha256"] == proof["provider_sha256"]
    assert raw["graph_sha256"] == model_proof["graph_sha256"] == sha(model / "model.onnx")
    assert raw["config_sha256"] == model_proof["config_sha256"]
    assert raw["provider"] == "webgpu" and raw["chunk"] == args.chunk
    assert raw["webgpu_max_pending_dispatches_override"] == 64
    assert raw["webgpu_validation_mode_override"] == ("full" if args.full_validation else None)
    assert raw["load_experiment_flags"]["ORT_WEBGPU_DENSE_BITCAST_UNPACK"] == "1"
    assert raw["load_experiment_flags"][flag] == "1"
    if args.candidate == "indexer-stack":
        assert raw["load_experiment_flags"]["ORT_WEBGPU_QSA_CONCAT_VECTOR"] == "0"
        assert raw["load_experiment_flags"]["ORT_WEBGPU_QSA_PACKED_SCORE"] == "0"
    assert raw["load_experiment_flags"]["ORT_WEBGPU_QMOE_EXTENDED_PREFILL"] == "1"
    controls = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())
    references = [s for s in controls["samples"] if s["context"] == args.context]
    assert len(raw["samples"]) == (2 if args.phase == "profile" else 4)
    tokens = 288 if args.phase == "windows" else 36 if args.phase == "profile" else 256
    for index, sample in enumerate(raw["samples"]):
        reference = references[1] if args.phase == "windows" else references[index]
        assert sample["prompt_sha256"] == reference["prompt_sha256"]
        assert len(sample["output_token_ids"]) == tokens
        assert sample["output_token_ids"][:256] == reference["output_token_ids"][:tokens]
        assert all(v["major_page_faults"] == 0 for v in sample["process_resource_usage"].values())
        if args.phase == "windows":
            assert sample["same_prompt_output_matches"]
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    if args.phase == "windows":
        assert raw["comparison"] == mode_prefix + "_decode_windows"
        assert raw["window_protocol"]["experiment_flag"] == flag
        result = verify_windows(raw, args.context, mode_prefix)
    elif args.phase == "standard":
        result = raw["summary"][f"{args.context}:configured"]
    else:
        profiles = list(ROOT.glob(prefix.name + f"-profile-ctx{args.context}-rep0-configured_*.json"))
        assert len(profiles) == 1, profiles
        programs = collections.Counter(event["name"].rsplit("&", 1)[-1] for event in json.loads(profiles[0].read_text())
                                       if event.get("cat") == "Kernel")
        program_names = ("QsaConcatVector4", "QsaPackedScore4") if args.candidate == "indexer-stack" else (program_name,)
        dispatched = {name: count for name, count in programs.items() if name in program_names}
        assert all(name in dispatched for name in program_names), programs
        result = dict(profile=str(profiles[0]), candidate_program_dispatches=dispatched,
                      instrumented_not_a_speed_claim=True)
    report = dict(status="original_sequences_qualified", context=args.context, phase=args.phase,
        candidate=args.candidate,
        full_validation=args.full_validation, chunk=args.chunk, zero_major_faults=True,
        original_output_prefixes_match=True, provider_sha256=proof["provider_sha256"],
        graph_sha256=model_proof["graph_sha256"], cache_preflight=cache,
        result=result, quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "context", "phase", "result")}), flush=True)


if __name__ == "__main__":
    main()
