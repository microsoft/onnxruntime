import argparse
import json
from pathlib import Path
import subprocess

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v3", "v4"), default="v3")
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--tokens", type=int, choices=(32, 256), default=32)
    parser.add_argument("--repetitions", type=int, choices=(1, 3), default=1)
    parser.add_argument("--modes", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    args = parser.parse_args()
    baseline_name = "expert-bitcast-unpack-v2-gpu7-standard2048-20261029"
    baseline = json.loads((ROOT / (baseline_name + ".qualification.json")).read_text())
    base_command = json.loads((ROOT / (baseline_name + ".guard.json")).read_text())["command"]
    control = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())
    references = [s for s in control["samples"] if s["context"] == args.context]
    assert len(references) >= args.repetitions + 1
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    provider = Path(f"/home/sayanshaw/repos/qwen38-int4-small-m-prefill/build-{args.version}/libonnxruntime_providers_webgpu.so")
    proof = json.loads((ROOT / f"small-m-prefill-{args.version}-native-tail-gpu7-20261029.qualification.json").read_text())
    assert sha(provider) == proof["provider_sha256"]
    assert proof["baseline_provider_sha256"] == baseline["provider_sha256"]
    assert sha(model / "model.onnx") == baseline["graph_sha256"]
    reports = []
    for mode in args.modes:
        prefix = ROOT / (f"small-m-prefill-{args.version}-mode{mode}-gpu7-screen{args.context}"
                         f"-t{args.tokens}-r{args.repetitions}-20261029")
        if any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json", ".diagnostic.json")):
            raise FileExistsError(prefix)
        command = base_command.copy()
        old_provider = "/workspace/qwen38-int4-expert-bitcast-unpack/build-v2/libonnxruntime_providers_webgpu.so"
        assert command[-1].count(old_provider) == 1
        command[-1] = command[-1].replace(old_provider, str(provider).replace(
            "/home/sayanshaw/repos", "/workspace")).replace(
            baseline_name, prefix.name).replace("--contexts 2048", f"--contexts {args.context}").replace(
            "--tokens 256 --repetitions 3",
            f"--tokens {args.tokens} --repetitions {args.repetitions} --chunk 16 --webgpu-validation-mode full")
        command[-1] = f"export ORT_WEBGPU_SMALL_M_PREFILL={mode}; " + command[-1]
        initial = snapshot()
        cache = prepare_model_cache(model, prefix)
        subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
            "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "3600",
            "--output-prefix", str(prefix), "--", *command], check=True)
        cache["after_run"] = snapshot_model_cache(model)
        Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
        guard = json.loads(prefix.with_suffix(".guard.json").read_text())
        raw = json.loads(prefix.with_suffix(".json").read_text())
        assert guard["status"] == raw["status"] == "completed" and guard["exit_code"] == 0
        assert guard["command"] == command and guard["gpu_index"] == 7
        assert raw["plugin_sha256"] == proof["provider_sha256"]
        assert raw["graph_sha256"] == baseline["graph_sha256"]
        assert raw["config_sha256"] == control["config_sha256"]
        assert raw["chunk"] == 16 and raw["provider"] == "webgpu"
        assert len(raw["samples"]) == args.repetitions + 1
        prompts_match = all(s["prompt_sha256"] == reference["prompt_sha256"]
                            for s, reference in zip(raw["samples"], references))
        outputs_match = all(s["output_token_ids"] == reference["output_token_ids"][:args.tokens]
                            for s, reference in zip(raw["samples"], references))
        no_faults = all(v["major_page_faults"] == 0 for s in raw["samples"]
                        for v in s["process_resource_usage"].values())
        log = prefix.with_suffix(".log").read_text()
        assert "GetCapability" not in log and "FAIL" not in log
        report = dict(status="small_m_prefill_screen", version=args.version, context=args.context, mode=mode,
            provider_sha256=proof["provider_sha256"], graph_sha256=baseline["graph_sha256"],
            original_prompts_match=prompts_match, original_output_prefixes_match=outputs_match,
            zero_major_faults=no_faults, full_validation=True, output_tokens=args.tokens,
            repetitions=args.repetitions, summary=raw["summary"][f"{args.context}:configured"],
            cache_preflight=cache, initial_resources=initial, final_resources=snapshot(),
            quality_qualified=False, headline_eligible=False)
        prefix.with_suffix(".diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        print(json.dumps({k: report[k] for k in ("status", "context", "mode",
            "original_output_prefixes_match", "zero_major_faults", "summary")}), flush=True)
        if not (prompts_match and outputs_match and no_faults):
            raise RuntimeError(f"Small-M screen failed original-prefix/fault gates: {prefix}")
    if len(reports) == 2:
        print("Both paired screen modes passed; diagnostic only, not whole-model quality qualification.")


if __name__ == "__main__":
    main()
