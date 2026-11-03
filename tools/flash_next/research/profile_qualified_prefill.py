import argparse
import collections
import json
from pathlib import Path
import subprocess

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_tokens


def profile_command(command, prefix, baseline_prefix):
    result = command.copy()
    assert result[-1].count("--tokens 256") == result[-1].count("--repetitions 3") == 1
    assert "--profile-prefix" not in result[-1] and "--profile-decode-after" not in result[-1]
    assert prefix.name != baseline_prefix and result[-1].count(baseline_prefix) == 3
    result[-1] = result[-1].replace(baseline_prefix, prefix.name)
    result[-1] = result[-1].replace("--tokens 256", "--tokens 2").replace("--repetitions 3", "--repetitions 1")
    result[-1] += f" --profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=32768)
    args = parser.parse_args()
    parent = json.loads((ROOT / "dense-block32-v2-contexts-20261030.qualification.json").read_text())
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    provider = Path("/home/sayanshaw/repos/qwen38-int4-dense-block32/build-v2/libonnxruntime_providers_webgpu.so")
    assert sha(provider) == parent["provider_sha256"]
    baseline = ROOT / f"dense-block32-v2-chunk1024-gpu7-full-standard{args.context}-20261030.guard.json"
    prefix = ROOT / f"qualified-stack-prefill-profile{args.context}-gpu7-20261031"
    if any(prefix.with_suffix(s).exists() for s in (".json", ".guard.json", ".qualification.json")):
        raise FileExistsError(prefix)
    command = profile_command(json.loads(baseline.read_text())["command"], prefix,
                              baseline.name.removesuffix(".guard.json"))
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
        "--output-prefix", str(prefix), "--", *command], check=True)
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw = json.loads(prefix.with_suffix(".json").read_text())
    guard = json.loads(prefix.with_suffix(".guard.json").read_text())
    verify_metadata(raw, guard, {"provider_sha256": parent["provider_sha256"]}, parent, True, chunk=1024)
    assert guard["command"] == command and raw["profiling_scope"] == "session"
    assert raw["graph_sha256"] == sha(model / "model.onnx")
    assert raw["config_sha256"] == sha(model / "genai_config.json")
    assert sha(provider) == parent["provider_sha256"]
    for flag in ("ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_DENSE_BLOCK32", "ORT_WEBGPU_DSA_QUERY_CACHE"):
        assert raw["load_experiment_flags"][flag] == "1"
    for flag in ("ORT_WEBGPU_QMOE_PREFILL2048", "ORT_WEBGPU_DENSE_STATIC_SHAPE",
                 "ORT_WEBGPU_QSA_PRUNED_RANK", "ORT_WEBGPU_QSA_SUBGROUP_RANK"):
        assert raw["load_experiment_flags"][flag] in (None, "0")
    references = [s for s in json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())["samples"]
                  if s["context"] == args.context]
    verify_tokens(raw, references, args.context, 2)
    profiles = list(ROOT.glob(prefix.name + "-profile_*.json"))
    assert len(profiles) == 1, profiles
    totals = collections.defaultdict(float)
    counts = collections.Counter()
    for event in json.loads(profiles[0].read_text()):
        if event.get("cat") == "Kernel":
            name = event["name"].rsplit("&", 1)[-1]
            totals[name] += event["dur"]
            counts[name] += 1
    assert totals
    log = prefix.with_suffix(".log").read_text()
    assert "FAIL" not in log and "GetCapability" not in log
    report = dict(status="baseline_prefill_profile_frozen_prefixes_qualified", context=args.context,
        full_validation=True, zero_major_faults=True, original_output_prefixes_match=True,
        provider_sha256=parent["provider_sha256"], graph_sha256=parent["graph_sha256"],
        config_sha256=parent["config_sha256"], chunk=1024,
        instrumented_session_kernel_totals_us=dict(totals), kernel_dispatches=dict(counts),
        instrumented_not_a_speed_claim=True, cache_preflight=cache,
        source_sha256={p.name: sha(p) for p in (*profiles, baseline, prefix.with_suffix(".json"),
                                              prefix.with_suffix(".guard.json"))},
        quality_qualified=False, headline_eligible=False)
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "context", "zero_major_faults")}), flush=True)


if __name__ == "__main__":
    main()
