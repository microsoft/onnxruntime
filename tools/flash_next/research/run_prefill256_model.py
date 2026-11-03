import argparse
import collections
import json
import math
from pathlib import Path
import subprocess

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_prefill128_dense import verify_samples
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_qmoe_prefill256_native import gate_rows


def verify_tokens(raw, references, context, tokens):
    assert len(raw["samples"]) == (4 if tokens == 256 else 2)
    assert [s["warmup"] for s in raw["samples"]] == [True] + [False] * (len(raw["samples"]) - 1)
    for sample, reference in zip(raw["samples"], references):
        assert sample["context"] == context
        assert sample["prompt_sha256"] == reference["prompt_sha256"]
        assert sample["output_tokens"] == len(sample["output_token_ids"]) == tokens
        assert sample["output_token_ids"] == reference["output_token_ids"][:tokens]
        assert all(v["major_page_faults"] == 0 for v in sample["process_resource_usage"].values())
        for field in ("decode_tps", "decode_seconds", "request_seconds", "ttft_seconds"):
            assert math.isfinite(sample[field]) and sample[field] > 0, (field, sample[field])
        assert math.isclose(sample["decode_tps"], (tokens - 1) / sample["decode_seconds"], rel_tol=1e-12)
    return verify_samples(raw, references, context) if tokens == 256 else raw["summary"][f"{context}:configured"]


def verify_metadata(raw, guard, native, parent, full_validation, chunk=256, pending=64):
    assert raw["status"] == guard["status"] == "completed" and guard["exit_code"] == 0
    assert guard["gpu_index"] == 7
    assert guard["gpu_uuid"] == "GPU-0e70e969-bbfc-80c5-d582-88be134632ea"
    assert raw["plugin_sha256"] == native["provider_sha256"]
    assert raw["graph_sha256"] == parent["graph_sha256"]
    assert raw["config_sha256"] == parent["config_sha256"]
    assert raw["provider"] == "webgpu" and raw["chunk"] == chunk
    assert isinstance(pending, int) and not isinstance(pending, bool) and pending > 0
    assert raw["webgpu_max_pending_dispatches_override"] == pending
    assert raw["webgpu_validation_mode_override"] == ("full" if full_validation else None)
    for flag, value in {"ORT_WEBGPU_QMOE_PREFILL256": "1", "ORT_WEBGPU_QMOE_EXTENDED_PREFILL": "1",
                        "ORT_WEBGPU_QSA_INDEXER_STACK": "1", "ORT_WEBGPU_DENSE_BITCAST_UNPACK": "1",
                        "ORT_WEBGPU_QSA_CONCAT_VECTOR": "0", "ORT_WEBGPU_QSA_PACKED_SCORE": "0"}.items():
        assert raw["load_experiment_flags"][flag] == value, flag


def verify_prefill_flags(raw, chunk, selector=None):
    assert chunk in (512, 1024, 1280, 2048)
    flags = ("ORT_WEBGPU_DENSE_TILE4", "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_QSA_SCORE16",
             "ORT_WEBGPU_QMOE_PREFILL512", selector or f"ORT_WEBGPU_QMOE_PREFILL{chunk}")
    if chunk in (1280, 2048):
        flags += ("ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DSA_QUERY_CACHE")
    if chunk == 2048:
        flags += ("ORT_WEBGPU_DENSE_BLOCK32",)
    assert all(raw["load_experiment_flags"][flag] == "1" for flag in flags)


def main(variant="prefill256-v1", stamp="20261029", parent_variant="qsa-indexer-stack-v1",
         parent_stamp="20261029", parent_chunk=128, selector="ORT_WEBGPU_QMOE_PREFILL256", chunk=256,
         parent_standard_prefix=None, parent_provider=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--phase", choices=("profile", "screen", "standard"), default="screen")
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.phase == "profile" and not args.full_validation:
        parser.error("Active-path profiling requires full validation")
    family, version = variant.rsplit("-v", 1)
    directory = Path(f"/home/sayanshaw/repos/qwen38-int4-qmoe-{family}/build-v{version}")
    provider = directory / "libonnxruntime_providers_webgpu.so"
    native = json.loads((ROOT / f"qmoe-{variant}-native-gpu7-{stamp}.qualification.json").read_text())
    parent = json.loads((ROOT / f"{parent_variant}-contexts-{parent_stamp}.qualification.json").read_text())
    assert native["status"] == f"prefill{chunk}_retained_chunk16_finite_bitwise_and_dispatch_qualified"
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert sha(provider) == native["provider_sha256"]
    if chunk in (512, 1024, 1280, 2048):
        assert native["finite_bitwise_comparisons"] == {
            512: 143523840, 1024: 300810240, 1280: 568197120, 2048: 615383040}[chunk]
        assert native["original_layers"] == 6
        assert native["rows_tested"] == {
            512: [16, 128, 256, 511, 512, 513],
            1024: [16, 128, 256, 512, 1023, 1024, 1025],
            1280: [16, 128, 256, 512, 768, 1024, 1279, 1280, 1281],
            2048: [16, 128, 256, 512, 1024, 2047, 2048, 2049]}[chunk]
        assert native["invalid_flag_explicitly_rejected"] and native["pending_windows_tested"] == [1, 64]
        fixture_version = "v4" if chunk == 512 else "v1"
        attention = json.loads((ROOT / f"{variant}-dsa-native-gpu7-fixtures{fixture_version}-{stamp}.qualification.json").read_text())
        assert attention["status"] == f"prefill{chunk}_dsa_cpu_reference_and_admission_qualified"
        assert attention["provider_sha256"] == native["provider_sha256"]
        assert attention["original_dsa_object_byte_identical"] and attention["expected_rejections"] == 12
        assert attention["default_sequence_bound_preserved"] and attention["invalid_selector_rejected"]
        if args.phase != "profile":
            proof = json.loads((ROOT / f"{variant}-gpu7-full-profile2048-{stamp}.qualification.json").read_text())
            assert proof["provider_sha256"] == native["provider_sha256"]
            assert proof["full_validation"] and proof["zero_major_faults"]
            assert proof["original_output_prefixes_match"]
            assert proof[f"active_prefill{chunk}_programs"] == {"QmoeGate1Token": 2 * (2048 // chunk) * 48}
    if not args.full_validation:
        validation_phase = "standard" if args.phase == "standard" else "screen"
        proof = json.loads((ROOT / (
            f"{variant}-gpu7-full-{validation_phase}{args.context}-{stamp}.qualification.json")).read_text())
        assert proof["provider_sha256"] == native["provider_sha256"] and proof["zero_major_faults"]
        assert proof["original_output_prefixes_match"] and proof["full_validation"]
        assert proof["context"] == args.context and proof["phase"] == validation_phase
    prefix = ROOT / (f"{variant}-gpu7-{'full-' if args.full_validation else ''}"
                     f"{args.phase}{args.context}-{stamp}")
    if prefix.with_suffix(".qualification.json").exists() or (
            not args.verify_existing and any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json"))):
        raise FileExistsError(prefix)
    baseline = parent_standard_prefix or f"{parent_variant}-chunk{parent_chunk}-gpu7-full-standard2048-{parent_stamp}"
    command = json.loads((ROOT / (baseline + ".guard.json")).read_text())["command"]
    parent_family, parent_version = parent_variant.rsplit("-v", 1)
    old_provider = (str(parent_provider).replace("/home/sayanshaw/repos", "/workspace")
                    if parent_provider is not None else
                    f"/workspace/qwen38-int4-{parent_family}/build-v{parent_version}/libonnxruntime_providers_webgpu.so")
    assert command[-1].count(old_provider) == command[-1].count(f"--chunk {parent_chunk}") == 1
    command[-1] = (f"export {selector}=1; " + command[-1].replace(
        old_provider, str(provider).replace("/home/sayanshaw/repos", "/workspace")).replace(
        baseline, prefix.name).replace("--contexts 2048", f"--contexts {args.context}").replace(
        f"--chunk {parent_chunk}", f"--chunk {chunk}"))
    tokens = 256 if args.phase == "standard" else 2 if args.phase == "profile" else 36
    marker = "--tokens 256 --repetitions 3"
    assert command[-1].count(marker) == 1
    command[-1] = command[-1].replace(marker, f"--tokens {tokens} --repetitions {3 if tokens == 256 else 1}")
    if not args.full_validation:
        assert command[-1].count(" --webgpu-validation-mode full") == 1
        command[-1] = command[-1].replace(" --webgpu-validation-mode full", "")
    if args.phase == "profile":
        command[-1] += f" --profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile"
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
    verify_metadata(raw, guard, native, parent, args.full_validation, chunk)
    assert raw["load_experiment_flags"][selector] == "1"
    if chunk in (512, 1024, 1280, 2048):
        verify_prefill_flags(raw, chunk)
    assert guard["command"] == command
    assert raw["graph_sha256"] == sha(model / "model.onnx")
    assert raw["config_sha256"] == sha(model / "genai_config.json")
    controls = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())
    references = [s for s in controls["samples"] if s["context"] == args.context]
    result = verify_tokens(raw, references, args.context, tokens)
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    coverage = None
    profiles = []
    if args.phase == "profile":
        profiles = list(ROOT.glob(prefix.name + "-profile_*.json"))
        assert len(profiles) == 1, profiles
        events = json.loads(profiles[0].read_text())
        coverage = collections.Counter(e["name"].rsplit("&", 1)[-1] for e in events
            if e.get("cat") == "Kernel" and e["name"].endswith("&QmoeGate1Token") and gate_rows(e) == chunk)
        assert coverage["QmoeGate1Token"] == 2 * (args.context // chunk) * 48, coverage
        if chunk == 1280:
            shapes = collections.Counter(gate_rows(e) for e in events
                if e.get("cat") == "Kernel" and e["name"].endswith("&QmoeGate1Token"))
            expected = {1280: 2 * (args.context // 1280) * 48}
            remainder = args.context % 1280
            if remainder:
                expected[remainder] = 2 * 48
            assert shapes == expected, shapes
    coverage_key = f"active_prefill{chunk}_programs"
    report = dict(status="original_sequences_qualified", context=args.context, phase=args.phase,
        full_validation=args.full_validation, chunk=chunk, zero_major_faults=True,
        original_output_prefixes_match=True, provider_sha256=native["provider_sha256"],
        baseline_provider_sha256=parent["provider_sha256"], graph_sha256=parent["graph_sha256"],
        config_sha256=parent["config_sha256"], **{coverage_key: coverage},
        result=result, cache_preflight=cache, source_sha256={p.name: sha(p) for p in [
            *profiles, prefix.with_suffix(".json"), prefix.with_suffix(".guard.json")]},
        quality_qualified=False, headline_eligible=False)
    prefix.with_suffix(".qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "context", "phase", "result",
                                           coverage_key)}), flush=True)


if __name__ == "__main__":
    main()
