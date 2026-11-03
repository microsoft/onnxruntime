import argparse
import json
from pathlib import Path
import re
import subprocess
import time

from build_qmoe_node_replay import replace_once, sha
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_gpu_greedy_argmax_ready_standard import BUILD as OLD_BUILD, command_for as old_command, verify_proof
from run_webgpu_int4_metrics import verify_result, verify_workers
from run_gpu_slice_toy_r2 import BUILD, PROVIDER, FLAG


VARIANT = "gpu-slice-v6-r2"


def prefix_for(context, full, flag, trial, trace):
    return ROOT / f"{VARIANT}-context{context}-{'full' if full else 'retained'}-flag{flag}-trial{trial}-trace{trace}-gpu7"


def verify_sources(path, expected):
    proof = json.loads(path.read_text())
    if proof["status"] != expected:
        raise RuntimeError(f"Unsupported prerequisite: {path}")
    for p, digest in proof["source_and_evidence_sha256"].items():
        resolved = host_path(p) if p.startswith(("/workspace/", "/evidence/")) else Path(p)
        if sha(resolved) != digest:
            raise RuntimeError(f"Changed evidence: {p}")
    return proof


def run(context, full, flag, trial, trace):
    prefix = prefix_for(context, full, flag, trial, trace)
    for suffix in (".json", ".guard.json", ".log", ".qualification.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    native_path = ROOT / "gpu-slice-v6-r2-toy-outcome.json"
    native = verify_sources(native_path, "aligned_gpu_slice_native_ids_scores_penalties_sampling_exact")
    manifests = [json.loads((p / "build-manifest.json").read_text()) for p in (BUILD, PROVIDER)]
    for manifest in manifests:
        for p, digest in manifest["source_sha256"].items():
            if sha(host_path(p)) != digest:
                raise RuntimeError(f"Changed build input: {p}")
    runtime_sha, provider_sha = manifests[0]["runtime_sha256"], manifests[1]["provider_sha256"]
    if (sha(BUILD / "libonnxruntime-genai.so") != runtime_sha or
            sha(PROVIDER / "libonnxruntime_providers_webgpu.so") != provider_sha or
            native["runtime_sha256"] != runtime_sha or native["provider_sha256"] != provider_sha):
        raise RuntimeError("Wrong qualified binaries")
    if context not in (2048, 8192, 32768) or (trace and (context != 2048 or not full)):
        raise ValueError("Invalid context or admission configuration")
    if trace == 0:
        verify_sources(ROOT / f"{VARIANT}-model-admission.json", "gpu_slice_frozen2k_prefixes_and_routing_admitted")
        verify_proof(ROOT / f"gpu-greedy-argmax-local8-v4-ready1-context{context}-repeat-screen.qualification.json",
                     "repeated_gpu_greedy_decode_gain_both_modes")
        if not full:
            verify_sources(prefix_for(context, True, flag, trial, 0).with_suffix(".qualification.json"),
                           "gpu_slice_frozen_original_prefixes_qualified")
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    parent = json.loads(parent_path.read_text())
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    if sha(model / "model.onnx") != parent["graph_sha256"] or sha(model / "genai_config.json") != parent["config_sha256"]:
        raise RuntimeError("Frozen model changed")
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [s for s in json.loads(controls_path.read_text())["samples"] if s["context"] == context]
    command = old_command(prefix, context, full, "gpu")
    old = "/workspace/" + str(OLD_BUILD.relative_to("/home/sayanshaw/repos"))
    new = "/workspace/" + str(BUILD.relative_to("/home/sayanshaw/repos"))
    command[-1] = command[-1].replace(old, new)
    command[-1] = replace_once(command[-1], new + "/libonnxruntime_providers_webgpu.so",
        "/workspace/" + str((PROVIDER / "libonnxruntime_providers_webgpu.so").relative_to("/home/sayanshaw/repos")))
    command[-1] = replace_once(command[-1], "benchmark_gpu_greedy_argmax_ready.py", "benchmark_gpu_slice.py")
    command[-1] = replace_once(command[-1], "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE=0",
                              f"ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE={trace}")
    command[-1] = (f"export {FLAG}={flag} ORT_WEBGPU_SLICE_UINT32_BITCOPY=1 "
                   "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH=0; " + command[-1])
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["free", "-h"], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader"], check=True)
    started = time.time_ns()
    subprocess.run(["python3", str(GUARD), "--gpu-index", "7", "--idle-seconds", "3",
        "--wait-seconds", "15", "--run-seconds", "2400", "--output-prefix", str(prefix), "--", *command], check=True)
    ended = time.time_ns()
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = [json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json")]
    flags = raw["load_experiment_flags"]
    expected_flags = {FLAG: str(flag), "ORT_WEBGPU_SLICE_UINT32_BITCOPY": "1",
        "ORT_GENAI_WEBGPU_GREEDY_ARGMAX": "1", "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE": str(trace),
        "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH": "0",
        "ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE": "1", "ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512": "0"}
    if (guard["command"] != command or raw["genai_native"]["sha256"] != runtime_sha or
            raw["plugin_sha256"] != provider_sha or raw["profiling"] or raw["comparison"] is not None or
            any(flags[k] != v for k, v in expected_flags.items())):
        raise RuntimeError("Wrong binary, command, or experiment configuration")
    if any(s["prefill_timing_scope"] != "append_tokens_through_first_token_selection" for s in raw["samples"]):
        raise RuntimeError("Wrong GPU-ready timing scope")
    workers = verify_workers(raw, "control")
    metrics = verify_result(raw, guard, dict(provider_sha256=provider_sha), parent, references, context, full)
    log = prefix.with_suffix(".log").read_text()
    sliced = re.findall(r"^WEBGPU_GPU_SLICE_COPY source_bytes=(\d+) offset_bytes=(\d+) copied_bytes=(\d+)$", log, re.M)
    if trace:
        expected = sum(s["output_tokens"] for s in raw["samples"])
        if (log.count("WEBGPU_GREEDY_ARGMAX rows=") != expected or
                log.count("WEBGPU_GREEDY_MIN_LENGTH rows=") != expected or
                "WEBGPU_GREEDY_ARGMAX_CPU_FALLBACK" in log or
                "WEBGPU_GREEDY_MIN_LENGTH_FUSED" in log):
            raise RuntimeError("Incomplete ordinary GPU greedy routing")
        if flag == 0 and sliced:
            raise RuntimeError("Disabled sliced copy executed")
        if flag == 1 and (len(sliced) != len(raw["samples"]) or
                any(int(count) != 248320 * 2 or int(offset) + int(count) != int(total)
                    for total, offset, count in sliced)):
            raise RuntimeError(f"Missing full-source last-row GPU copy admission: {sliced}")
    elif "WEBGPU_GPU_SLICE_COPY " in log or "WEBGPU_GREEDY_ARGMAX rows=" in log:
        raise RuntimeError("Diagnostic trace contaminated natural timing")
    sources = [Path(__file__), ROOT / "benchmark_gpu_slice.py", GUARD, native_path,
        BUILD / "build-manifest.json", PROVIDER / "build-manifest.json", parent_path, controls_path,
        BUILD / "libonnxruntime-genai.so", PROVIDER / "libonnxruntime_providers_webgpu.so",
        Path(cache["evidence_path"]), *[prefix.with_suffix(s) for s in (".json", ".guard.json", ".log")]]
    result = dict(status="gpu_slice_frozen_original_prefixes_qualified", context=context, full_validation=full,
        flag=flag, trace=trace, trial=trial, runtime_sha256=runtime_sha, provider_sha256=provider_sha,
        original_output_prefixes_match=True, started_ns=started, completed_ns=ended, workers=workers,
        gpu_slice_calls=len(sliced), slice_transfers=sliced, metrics=metrics,
        natural_speedup_claimed=False, quality_qualified=False,
        source_and_evidence_sha256={str(p): sha(p) for p in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "flag", "trace", "gpu_slice_calls", "metrics")}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", type=int, choices=(0, 1), required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--trace", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    run(2048, True, args.flag, args.trial, args.trace)
