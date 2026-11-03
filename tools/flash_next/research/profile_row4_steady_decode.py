import argparse
import collections
import json
import math
from pathlib import Path
import subprocess

from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_prefill_flags
from run_qmoe_prefill_rows4_standard import DIRECTORY, TEMPLATE, standard_command


def profile_command(template, provider, prefix, context):
    command = standard_command(template, provider, prefix, context, True)
    assert command[-1].count("--tokens 256 --repetitions 3") == 1
    assert "--profile-" not in command[-1]
    command[-1] = command[-1].replace("--repetitions 3", "--repetitions 1")
    command[-1] += (f" --profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile"
                   " --profile-decode-after 255")
    return command


def verify_profile_samples(raw, references, context):
    assert len(raw["samples"]) == 2
    assert [sample["warmup"] for sample in raw["samples"]] == [True, False]
    for sample, reference in zip(raw["samples"], references):
        assert sample["context"] == context and sample["prompt_sha256"] == reference["prompt_sha256"]
        assert sample["output_tokens"] == len(sample["output_token_ids"]) == 256
        assert sample["output_token_ids"] == reference["output_token_ids"][:256]
        assert sample["profile_after_tokens"] == 255 and sample["profiled_decode_tokens"] == 1
        assert all(phase["major_page_faults"] == 0 for phase in sample["process_resource_usage"].values())
        assert all(math.isfinite(sample[field]) and sample[field] > 0 for field in
                   ("decode_seconds", "decode_tps", "request_seconds", "ttft_seconds"))
        assert math.isclose(sample["decode_tps"] * sample["decode_seconds"], 255, rel_tol=1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", choices=(2048, 8192, 32768), type=int, default=32768)
    args = parser.parse_args()
    prefix = ROOT / f"row4-steady-decode-profile255-context{args.context}-gpu7-20261101"
    for suffix in (".json", ".guard.json", ".log", ".qualification.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    native_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
    parent, native = (json.loads(path.read_text()) for path in (parent_path, native_path))
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == parent["provider_sha256"] == native["provider_sha256"]
    template_path = ROOT / (TEMPLATE + ".guard.json")
    command = profile_command(json.loads(template_path.read_text())["command"], provider, prefix, args.context)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
        "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
        "--output-prefix", str(prefix), "--", *command], check=True)
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = (json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json"))
    verify_metadata(raw, guard, native, parent, True, chunk=1024)
    verify_prefill_flags(raw, 1024)
    assert guard["command"] == command and raw["profiling_scope"] == "decode_only"
    assert raw["profiling"] and raw["cpu_affinity"] == list(range(96))
    assert "worker_affinity_override" not in raw
    references_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [sample for sample in json.loads(references_path.read_text())["samples"]
                  if sample["context"] == args.context]
    verify_profile_samples(raw, references, args.context)
    profiles = []
    results = []
    for sample in raw["samples"]:
        matches = list(ROOT.glob(Path(sample["decode_profile_prefix"]).name + "_*.json"))
        assert len(matches) == 1, matches
        profiles.extend(matches)
        counts, durations = collections.Counter(), collections.defaultdict(float)
        for event in json.loads(matches[0].read_text()):
            if event.get("cat") == "Kernel":
                name = event["name"].rsplit("&", 1)[-1]
                assert math.isfinite(event["dur"]) and event["dur"] >= 0
                counts[name] += 1
                durations[name] += event["dur"]
        assert durations
        results.append(dict(warmup=sample["warmup"], kernel_counts=dict(counts),
                            instrumented_kernel_totals_us=dict(durations)))
    sources = [Path(__file__), parent_path, native_path, template_path, references_path, *profiles,
               *[prefix.with_suffix(s) for s in (".json", ".guard.json", ".log")]]
    report = dict(status="steady_single_decode_frozen_512_ids_qualified_not_tps",
        context=args.context, profile_after_tokens=255, profiled_decode_tokens_per_request=1,
        original_output_prefixes_match=True, zero_major_faults=True,
        provider_sha256=parent["provider_sha256"], graph_sha256=parent["graph_sha256"],
        config_sha256=parent["config_sha256"], results=results, cache_preflight=cache,
        instrumented_not_a_speed_claim=True, quality_qualified=False, headline_eligible=False,
        source_sha256={str(path): sha(path) for path in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "context", "results")}), flush=True)


if __name__ == "__main__":
    main()
