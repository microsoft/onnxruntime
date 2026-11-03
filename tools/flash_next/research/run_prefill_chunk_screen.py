import argparse
import json
from pathlib import Path
import subprocess

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--chunks", type=int, nargs="+", choices=(16, 32, 64, 128, 256),
                        default=[16, 64])
    parser.add_argument("--extended-prefill", action="store_true")
    parser.add_argument("--dense-stack", action="store_true")
    parser.add_argument("--tokens", type=int, choices=(32, 256), default=32)
    parser.add_argument("--repetitions", type=int, choices=(1, 3), default=1)
    parser.add_argument("--standard-validation", action="store_true",
                        help="Keep the retained benchmark's validation setting after full-validation proof.")
    args = parser.parse_args()
    if args.dense_stack and not args.extended_prefill:
        parser.error("--dense-stack requires --extended-prefill")
    if args.standard_validation and args.extended_prefill and (
            args.chunks not in ([64], [128]) or args.tokens != 256 or args.repetitions != 3):
        parser.error("Extended standard-validation runs require one chunk of 64 or 128, "
                     "--tokens 256 --repetitions 3")
    variant = ("prefill-dense-stack-v1" if args.dense_stack else
               "qmoe-extended-prefill-v1" if args.extended_prefill else "prefill")
    baseline_name = "expert-bitcast-unpack-v2-gpu7-standard2048-20261029"
    baseline = json.loads((ROOT / (baseline_name + ".qualification.json")).read_text())
    base_command = json.loads((ROOT / (baseline_name + ".guard.json")).read_text())["command"]
    control = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())
    references = [s for s in control["samples"] if s["context"] == args.context]
    assert len(references) >= args.repetitions + 1
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    provider = Path("/home/sayanshaw/repos/qwen38-int4-expert-bitcast-unpack/build-v2/libonnxruntime_providers_webgpu.so")
    if args.extended_prefill:
        proof = json.loads((ROOT / "qmoe-extended-prefill-v1-native-gpu7-20261029.qualification.json").read_text())
        provider = Path("/home/sayanshaw/repos/qwen38-int4-qmoe-extended-prefill/build-v1/libonnxruntime_providers_webgpu.so")
        assert proof["baseline_provider_sha256"] == baseline["provider_sha256"]
        provider_sha = proof["provider_sha256"]
        if args.dense_stack:
            stack = json.loads((ROOT / "prefill-dense-stack-v1-native-gpu7-20261029.qualification.json").read_text())
            assert stack["status"] == "combined_dense_and_prefill_native_qualified"
            assert stack["baseline_provider_sha256"] == provider_sha
            provider = Path("/home/sayanshaw/repos/qwen38-int4-prefill-dense-stack/build-v1/libonnxruntime_providers_webgpu.so")
            provider_sha = stack["provider_sha256"]
        if args.standard_validation:
            for context in (2048, 8192, 32768):
                full = ROOT / (f"{variant}-chunk{args.chunks[0]}-gpu7-screen{context}"
                               "-t256-r3-20261029.diagnostic.json")
                evidence = json.loads(full.read_text())
                assert evidence["provider_sha256"] == provider_sha
                assert evidence["context"] == context and evidence["chunk"] == args.chunks[0]
                assert evidence["original_prompts_match"] and evidence["original_output_prefixes_match"]
                assert evidence["zero_major_faults"]
                assert evidence["full_validation"] and evidence["output_tokens"] == 256
                assert evidence["repetitions"] == 3
    else:
        provider_sha = baseline["provider_sha256"]
    assert sha(provider) == provider_sha
    assert sha(model / "model.onnx") == baseline["graph_sha256"]
    for chunk in args.chunks:
        suffix = "" if (args.tokens, args.repetitions) == (32, 1) else f"-t{args.tokens}-r{args.repetitions}"
        if args.standard_validation:
            suffix += "-standard-validation"
        prefix = ROOT / f"{variant}-chunk{chunk}-gpu7-screen{args.context}{suffix}-20261029"
        if any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json", ".qualification.json")):
            raise FileExistsError(prefix)
        command = base_command.copy()
        assert "--tokens 256 --repetitions 3" in command[-1]
        validation = "" if args.standard_validation else " --webgpu-validation-mode full"
        command[-1] = command[-1].replace(baseline_name, prefix.name).replace(
            "--contexts 2048", f"--contexts {args.context}").replace(
            "--tokens 256 --repetitions 3",
            f"--tokens {args.tokens} --repetitions {args.repetitions} --chunk {chunk}{validation}")
        if args.extended_prefill:
            original_provider = "/workspace/qwen38-int4-expert-bitcast-unpack/build-v2/libonnxruntime_providers_webgpu.so"
            assert command[-1].count(original_provider) == 1
            flags = "ORT_WEBGPU_QMOE_EXTENDED_PREFILL=1"
            if args.dense_stack:
                flags += " ORT_WEBGPU_DENSE_BITCAST_UNPACK=1"
            command[-1] = f"export {flags}; " + command[-1].replace(
                original_provider, str(provider).replace("/home/sayanshaw/repos", "/workspace"))
        initial = snapshot()
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
        assert raw["plugin_sha256"] == provider_sha
        assert raw["graph_sha256"] == baseline["graph_sha256"]
        assert raw["config_sha256"] == control["config_sha256"]
        assert raw["chunk"] == chunk and raw["provider"] == "webgpu"
        assert raw["webgpu_validation_mode_override"] == (None if args.standard_validation else "full")
        if args.dense_stack:
            assert raw["load_experiment_flags"]["ORT_WEBGPU_DENSE_BITCAST_UNPACK"] == "1"
            assert raw["load_experiment_flags"]["ORT_WEBGPU_QMOE_EXTENDED_PREFILL"] == "1"
        assert len(raw["samples"]) == args.repetitions + 1
        prompts_match = all(s["prompt_sha256"] == reference["prompt_sha256"]
                            for s, reference in zip(raw["samples"], references))
        outputs_match = all(s["output_token_ids"] == reference["output_token_ids"][:args.tokens]
                            for s, reference in zip(raw["samples"], references))
        no_faults = all(v["major_page_faults"] == 0 for s in raw["samples"]
                        for v in s["process_resource_usage"].values())
        log = prefix.with_suffix(".log").read_text()
        assert "GetCapability" not in log and "FAIL" not in log
        report = dict(status="prefill_chunk_diagnostic", context=args.context, chunk=chunk,
            provider_sha256=provider_sha, graph_sha256=baseline["graph_sha256"],
            extended_prefill=args.extended_prefill,
            dense_stack=args.dense_stack,
            original_prompts_match=prompts_match, original_output_prefixes_match=outputs_match,
            original_32_token_prefixes_match=all(
                s["output_token_ids"][:32] == reference["output_token_ids"][:32]
                for s, reference in zip(raw["samples"], references)),
            zero_major_faults=no_faults, full_validation=not args.standard_validation,
            standard_validation=args.standard_validation, output_tokens=args.tokens,
            repetitions=args.repetitions,
            summary=raw["summary"][f"{args.context}:configured"],
            cache_preflight=cache, initial_resources=initial, final_resources=snapshot(),
            quality_qualified=False, headline_eligible=False)
        prefix.with_suffix(".diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("status", "context", "chunk",
            "original_output_prefixes_match", "zero_major_faults", "summary")}), flush=True)
        if not (prompts_match and outputs_match and no_faults):
            raise RuntimeError(f"Prefill chunk screen failed original-prefix/fault gates: {prefix}")


if __name__ == "__main__":
    main()
