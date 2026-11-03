import argparse
import json
from pathlib import Path
import subprocess

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_qsa_score16_standard import verify_standard
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_prefill_flags
from run_qmoe_prefill_rows4_standard import DIRECTORY, TEMPLATE, standard_command


def pending_command(template, provider, prefix, context, full, pending):
    assert pending in (8, 16, 32, 64)
    command = standard_command(template, provider, prefix, context, full)
    marker = "--webgpu-max-pending-dispatches 64"
    assert command[-1].count(marker) == 1
    command[-1] = command[-1].replace(marker, f"--webgpu-max-pending-dispatches {pending}")
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.context != 2048:
        parser.error("Long-context tuning requires a separately qualified both-mode 2K screen first")
    prefix = args.output_prefix
    for suffix in (".json", ".guard.json", ".qualification.json", ".log"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    native_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    native, parent = (json.loads(path.read_text()) for path in (native_path, parent_path))
    assert native["finite_bitwise_comparisons"] == 300810240
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == native["provider_sha256"] == parent["provider_sha256"]
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    command = pending_command(json.loads(template_path.read_text())["command"], provider, prefix,
                              args.context, args.full_validation, args.pending)
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "900",
                    "--output-prefix", str(prefix), "--", *command], check=True)
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = (json.loads(prefix.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json"))
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [sample for sample in json.loads(controls_path.read_text())["samples"]
                  if sample["context"] == args.context]
    result = verify_standard(raw, guard, native, parent, references, args.context, args.full_validation,
                             chunk=1024, pending=args.pending)
    verify_prefill_flags(raw, 1024)
    assert raw["load_experiment_flags"]["ORT_WEBGPU_QMOE_GROUPED_PREFILL"] == "1"
    assert raw["load_experiment_flags"]["ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD"] == "0"
    assert guard["command"] == command
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    report = dict(status="original_sequences_qualified", phase="standard", context=args.context,
                  full_validation=args.full_validation, pending=args.pending, chunk=1024,
                  provider_sha256=native["provider_sha256"], graph_sha256=parent["graph_sha256"],
                  config_sha256=parent["config_sha256"], result=result, zero_major_faults=True,
                  original_output_prefixes_match=True, original_sequence_comparisons=1024,
                  comparison_kind="separate_load_tuning_observation_not_matched_causal_evidence",
                  measured_repetitions=3, excluded_warmup_requests=1, output_tokens=256,
                  quality_qualified=False, headline_eligible=False,
                  source_sha256={str(path): sha(path) for path in
                                 (prefix.with_suffix(".json"), prefix.with_suffix(".guard.json"),
                                  Path(__file__), native_path, parent_path, template_path, controls_path)})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "context", "pending", "result")}), flush=True)


if __name__ == "__main__":
    main()
