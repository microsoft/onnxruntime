import argparse
import json
from pathlib import Path
import subprocess

from build_genai_allocator_single_thread import FLAG
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from qualify_quickgelu_mul_long_context import ROOT, sha
from run_prefill256_model import verify_metadata, verify_tokens
from run_qmoe_prefill_rows4_standard import DIRECTORY, TEMPLATE, standard_command


RUNTIME = Path("/home/sayanshaw/repos/qwen38-int4-genai-allocator-single-thread/build-v1")
OLD_RUNTIME = "/workspace/onnxruntime-genai-webgpu-fp8-checkpoint/build-h200-int4-legacy"


def worker_command(template, provider, prefix, mode):
    assert mode in ("control", "single")
    command = standard_command(template, provider, prefix, 2048, True)
    assert command[-1].count(OLD_RUNTIME) == 1
    assert command[-1].count("--tokens 256 --repetitions 3") == 1
    command[-1] = (f"export {FLAG}={int(mode == 'single')}; " + command[-1].replace(
        OLD_RUNTIME, str(RUNTIME).replace("/home/sayanshaw/repos", "/workspace")).replace(
            "--tokens 256 --repetitions 3", "--tokens 36 --repetitions 1"))
    return command


def verify_workers(raw, mode):
    assert mode in ("control", "single")
    assert raw["cpu_affinity"] == list(range(96)) and "worker_affinity_override" not in raw
    masks = raw["thread_cpu_affinity"]
    assert masks and all(mask and set(mask) <= set(range(96)) for mask in masks.values())
    pinned = [mask for mask in masks.values() if len(mask) == 1]
    if mode == "control":
        assert len(pinned) == 95 and {tuple(mask) for mask in pinned} == {(cpu,) for cpu in range(1, 96)}
    else:
        assert not pinned and all(mask == list(range(96)) for mask in masks.values())
    return dict(total_observed_threads=len(masks), individually_pinned_threads=len(pinned))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("control", "single"), required=True)
    args = parser.parse_args()
    prefix = ROOT / f"genai-allocator-single-thread-v1-{args.mode}-workers-gpu7-20261031"
    for suffix in (".json", ".guard.json", ".log", ".qualification.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    build_path = RUNTIME / "build-manifest.json"
    build = json.loads(build_path.read_text())
    assert build["baseline_relink_byte_identical"] and build["baseline_rebuild_byte_identical"]
    runtime = RUNTIME / "libonnxruntime-genai.so"
    assert sha(runtime) == build["runtime_sha256"]
    for name, digest in build["source_sha256"].items():
        path = Path(name.replace("/workspace", "/home/sayanshaw/repos", 1).replace(
            "/evidence", str(ROOT.parent), 1))
        assert sha(path) == digest, name
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    native_path = ROOT / "qmoe-prefill-rows4-v2-native-gpu7-20261031.qualification.json"
    parent, native = (json.loads(path.read_text()) for path in (parent_path, native_path))
    assert parent["all_original_sequences_match"] and parent["zero_major_faults"]
    provider = DIRECTORY / "libonnxruntime_providers_webgpu.so"
    assert sha(provider) == parent["provider_sha256"] == native["provider_sha256"]
    template_path = ROOT / f"{TEMPLATE}.guard.json"
    command = worker_command(json.loads(template_path.read_text())["command"], provider, prefix, args.mode)
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    assert sha(model / "model.onnx") == parent["graph_sha256"]
    assert sha(model / "genai_config.json") == parent["config_sha256"]
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["python3", str(ROOT / "shared_activation_guard.py"), "--gpu-index", "7",
                    "--idle-seconds", "3", "--wait-seconds", "15", "--run-seconds", "2400",
                    "--output-prefix", str(prefix), "--", *command], check=True)
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = (json.loads(prefix.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json"))
    assert guard["command"] == command and raw["genai_native"]["sha256"] == build["runtime_sha256"]
    verify_metadata(raw, guard, native, parent, True, chunk=1024)
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [sample for sample in json.loads(controls_path.read_text())["samples"] if sample["context"] == 2048]
    verify_tokens(raw, references, 2048, 36)
    assert raw["comparison"] is None and not raw["profiling"]
    workers = verify_workers(raw, args.mode)
    sources = [Path(__file__), ROOT / "benchmark_webgpu_int4.py", parent_path, native_path, build_path,
               runtime, template_path, controls_path, *[prefix.with_suffix(suffix) for suffix in
                                                        (".json", ".guard.json", ".log")]]
    report = dict(status="allocator_worker_policy_original_prefixes_qualified", mode=args.mode,
                  runtime_sha256=build["runtime_sha256"], provider_sha256=native["provider_sha256"],
                  workers=workers, original_output_prefixes_match=True, zero_major_faults=True,
                  resource_diagnostic_only=True, standard_tps_qualified=False,
                  main_affinity_unchanged=True, manual_affinity_override=False,
                  model_and_arithmetic_unchanged=True, quality_qualified=False, headline_eligible=False,
                  cache_preflight=cache, source_sha256={str(path): sha(path) for path in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "mode", "workers")}), flush=True)


if __name__ == "__main__":
    main()
