import argparse
import json
from pathlib import Path
import subprocess

from build_qsa_prefill_score16 import FLAG
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_qsa_score16_standard import verify_standard
from qualify_quickgelu_mul_long_context import ROOT, sha


VARIANT = "qsa-prefill-score16-v1"
PROVIDER = Path("/home/sayanshaw/repos/qwen38-int4-qsa-prefill-score16/build-v1/libonnxruntime_providers_webgpu.so")
OLD_PROVIDER = "/workspace/qwen38-int4-dense-block32/build-v2/libonnxruntime_providers_webgpu.so"


def screen_gate(screens, native, parent):
    assert len(screens) == 6
    assert {(r["context"], r["full_validation"]) for r in screens} == {
        (context, full) for context in (2048, 8192, 32768) for full in (True, False)}
    assert native["baseline_provider_sha256"] == parent["provider_sha256"]
    for row in screens:
        assert row["provider_sha256"] == native["provider_sha256"]
        assert row["baseline_provider_sha256"] == parent["provider_sha256"]
        assert row["graph_sha256"] == parent["graph_sha256"] and row["config_sha256"] == parent["config_sha256"]
        assert row["all_original_sequences_match"] and row["repeated_sequences_match"] and row["zero_major_faults"]
        assert row["original_sequence_comparisons"] == 288 and row["output_tokens"] == 36
        assert row["order"] == [0, 1, 1, 0] and row["measured_requests"] == 4
        assert row["status"] in ("paired_ttft_win_qualified", "exact_sequences_qualified_no_paired_ttft_win")
        if row["context"] != 2048 or row["full_validation"]:
            assert row["ttft_win_screen_qualified"] and row["all_pairs_favor_candidate"]
            assert row["geometric_mean_matched_ttft_ratio"] > 1
    retained2k = next(r for r in screens if r["context"] == 2048 and not r["full_validation"])
    return dict(long_contexts_faster_in_both_validation_modes=True,
                retained_2k_screen_ttft_win=retained2k["ttft_win_screen_qualified"],
                retained_2k_matched_ttft_reduction_percent=retained2k["matched_ttft_reduction_percent"])


def prerequisites():
    native_path = ROOT / f"{VARIANT}-native-gpu7-20261031.qualification.json"
    parent_path = ROOT / "dense-block32-v2-contexts-20261030.qualification.json"
    native, parent = (json.loads(p.read_text()) for p in (native_path, parent_path))
    assert native["status"] == "prefill_score16_exact_reference_outputs_and_admission_qualified"
    assert native["fixtures"] == 76 and native["reference_executions"] == 2736
    assert native["invalid_selector_rejected"] and native["original_shader_and_decode_admission_preserved"]
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    assert sha(PROVIDER) == native["provider_sha256"]
    paths = [ROOT / f"{VARIANT}-gpu7-{'full-' if full else ''}paired-screen{context}-20261031.qualification.json"
             for context in (2048, 8192, 32768) for full in (True, False)]
    screens = [json.loads(p.read_text()) for p in paths]
    gate = screen_gate(screens, native, parent)
    profile_path = ROOT / f"{VARIANT}-mode1-gpu7-full-profile2048-20261031.qualification.json"
    profile = json.loads(profile_path.read_text())
    assert profile["provider_sha256"] == native["provider_sha256"] and profile["full_validation"]
    assert profile["active_prefill_score16_programs"] == 48
    assert profile["zero_major_faults"] and profile["original_output_prefixes_match"]
    sources = {p.name: sha(p) for p in (native_path, parent_path, profile_path, *paths)}
    for screen in screens:
        for name, digest in screen["source_sha256"].items():
            assert sha(ROOT / name) == digest
            sources[name] = digest
    return native, parent, gate, sources


def standard_command(command, baseline, prefix, provider, full):
    result = list(command)
    assert result[-1].count(OLD_PROVIDER) == 1 and result[-1].count(baseline) == 3
    assert result[-1].count("--tokens 256 --repetitions 3") == 1
    result[-1] = (f"export {FLAG}=1; " + result[-1].replace(
        OLD_PROVIDER, str(provider).replace("/home/sayanshaw/repos", "/workspace")).replace(baseline, prefix))
    if not full:
        assert result[-1].count(" --webgpu-validation-mode full") == 1
        result[-1] = result[-1].replace(" --webgpu-validation-mode full", "")
    return result


def verify_case(raw, guard, native, parent, references, context, full):
    assert raw["load_experiment_flags"][FLAG] == "1"
    for flag in ("ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DSA_QUERY_CACHE", "ORT_WEBGPU_DENSE_BLOCK32"):
        assert raw["load_experiment_flags"][flag] == "1"
    assert raw["load_experiment_flags"]["ORT_WEBGPU_DENSE_DIRECT_A"] in (None, "0")
    assert raw["load_experiment_flags"]["ORT_WEBGPU_DENSE_PRIVATE_PARTIALS"] in (None, "0")
    return verify_standard(raw, guard, native, parent, references, context, full, chunk=1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), required=True)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    native, parent, gate, sources = prerequisites()
    prefix = ROOT / f"{VARIANT}-chunk1024-gpu7-{'full-' if args.full_validation else ''}standard{args.context}-20261031"
    if prefix.with_suffix(".qualification.json").exists() or (
            not args.verify_existing and any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json"))):
        raise FileExistsError(prefix)
    if not args.full_validation:
        full_path = ROOT / f"{VARIANT}-chunk1024-gpu7-full-standard{args.context}-20261031.qualification.json"
        proof = json.loads(full_path.read_text())
        assert proof["provider_sha256"] == native["provider_sha256"]
        assert proof["full_validation"] and proof["original_output_prefixes_match"] and proof["zero_major_faults"]
        sources[full_path.name] = sha(full_path)
    baseline = f"dense-block32-v2-chunk1024-gpu7-full-standard{args.context}-20261030"
    baseline_path = ROOT / (baseline + ".guard.json")
    command = standard_command(json.loads(baseline_path.read_text())["command"],
                               baseline, prefix.name, PROVIDER, args.full_validation)
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
    raw, guard = (json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json"))
    assert guard["command"] == command
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [s for s in json.loads(controls_path.read_text())["samples"] if s["context"] == args.context]
    result = verify_case(raw, guard, native, parent, references, args.context, args.full_validation)
    assert sha(PROVIDER) == native["provider_sha256"]
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    for path in (baseline_path, controls_path, Path(__file__), *(prefix.with_suffix(s) for s in (".json", ".guard.json", ".log"))):
        sources[path.name] = sha(path)
    report = dict(status="original_sequences_qualified", context=args.context, phase="standard",
        full_validation=args.full_validation, chunk=1024, result=result, screen_gate=gate,
        zero_major_faults=True, original_output_prefixes_match=True, cache_preflight=cache,
        provider_sha256=native["provider_sha256"], baseline_provider_sha256=parent["provider_sha256"],
        graph_sha256=parent["graph_sha256"], config_sha256=parent["config_sha256"],
        quality_qualified=False, provider_promoted=False, source_sha256=sources)
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "context", "full_validation", "result")}), flush=True)


if __name__ == "__main__":
    main()
