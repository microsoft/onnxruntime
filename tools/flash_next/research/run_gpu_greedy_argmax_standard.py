import argparse
import json
from pathlib import Path
import subprocess
import time

from build_qmoe_node_replay import sha
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_gpu_greedy_argmax_initial_model_v4 import BUILD, FLAG
from run_webgpu_int4_metrics import RUNTIME, TEMPLATE, model_command, verify_result, verify_workers


VARIANT = "gpu-greedy-argmax-local8-v4"
BENCHMARK = ROOT / "benchmark_gpu_greedy_argmax_natural.py"


def prefix_for(mode, context, full, trial):
    return ROOT / f"{VARIANT}-{mode}-gpu7-{'full-' if full else ''}standard{context}-trial{trial}"


def verify_proof(path, expected_status):
    proof = json.loads(path.read_text())
    if proof["status"] != expected_status:
        raise RuntimeError(f"Unsupported prerequisite: {path}")
    digests = proof["source_sha256"] if "source_sha256" in proof else proof["source_and_evidence_sha256"]
    for path, digest in digests.items():
        if sha(Path(path)) != digest:
            raise RuntimeError(f"Changed prerequisite: {path}")
    return proof


def command_for(prefix, context, full, mode):
    if mode not in ("control", "gpu"):
        raise ValueError(mode)
    template = json.loads((ROOT / f"{TEMPLATE}.guard.json").read_text())["command"]
    command = model_command(template, BUILD / "libonnxruntime_providers_webgpu.so", prefix, context, full, "columns8")
    runtime = "/workspace/" + str(RUNTIME.relative_to("/home/sayanshaw/repos"))
    old_benchmark = "/evidence/webgpu-perf-oct20/benchmark_webgpu_int4.py"
    if command[-1].count(runtime) != 1 or command[-1].count(old_benchmark) != 1:
        raise RuntimeError("Nonunique supported-runtime/benchmark replacement")
    command[-1] = command[-1].replace(runtime, "/workspace/" + str(BUILD.relative_to("/home/sayanshaw/repos")))
    command[-1] = command[-1].replace(old_benchmark, "/evidence/webgpu-perf-oct20/" + BENCHMARK.name)
    command[-1] = (f"export {FLAG}={int(mode == 'gpu')} ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE=0 "
                   "ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE=1 ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512=0; " + command[-1])
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("control", "gpu"), required=True)
    parser.add_argument("--context", type=int, choices=(2048, 8192, 32768), default=2048)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--trial", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.mode == "gpu" and args.trial == 2:
        parser.error("Trial2 is reserved for the ending control")
    prefix = prefix_for(args.mode, args.context, args.full_validation, args.trial)
    for suffix in (".json", ".guard.json", ".log", ".qualification.json"):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError(prefix.with_suffix(suffix))
    initial_path = ROOT / "gpu-greedy-argmax-v4-initial-model-outcome.json"
    initial = verify_proof(initial_path, "gpu_greedy_argmax_default_selected_prefixes_and_routing_qualified")
    primitive_path = ROOT / "argmax-parallel-bitwise-v4-followup-outcome.json"
    mask_path = ROOT / "gpu-greedy-argmax-v4-mask-native-outcome.json"
    verify_proof(primitive_path, "primitive_followup_completed")
    verify_proof(mask_path, "gpu_min_length_cast_where_argmax_cpu_ids_exact")
    manifest_path = BUILD / "build-manifest.json"
    build = json.loads(manifest_path.read_text())
    for path, digest in build["source_sha256"].items():
        if sha(host_path(path)) != digest:
            raise RuntimeError(f"Changed private runtime source: {path}")
    for filename, key in (("libonnxruntime-genai.so", "runtime_sha256"),
                          ("libonnxruntime_providers_webgpu.so", "provider_sha256")):
        if sha(BUILD / filename) != build[key] or initial[key] != build[key]:
            raise RuntimeError("Changed qualified binary")
    prerequisites = []
    if not args.full_validation:
        prerequisites.append((prefix_for(args.mode, args.context, True, args.trial).with_suffix(".qualification.json"),
                              "gpu_greedy_original_prefixes_qualified"))
    if args.mode == "gpu" and args.trial == 1:
        prerequisites.append((ROOT / f"{VARIANT}-context{args.context}-screen.qualification.json",
                              "initial_gpu_greedy_decode_gain_both_modes"))
    if args.context != 2048:
        previous = 2048 if args.context == 8192 else 8192
        prerequisites.append((ROOT / f"{VARIANT}-context{previous}-repeat-screen.qualification.json",
                              "repeated_gpu_greedy_decode_gain_both_modes"))
    for path, expected in prerequisites:
        proof = verify_proof(path, expected)
        if proof["runtime_sha256"] != build["runtime_sha256"] or proof["provider_sha256"] != build["provider_sha256"]:
            raise RuntimeError("Wrong context/mode prerequisite binary")
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    parent = json.loads(parent_path.read_text())
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [x for x in json.loads(controls_path.read_text())["samples"] if x["context"] == args.context]
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    if sha(model / "model.onnx") != parent["graph_sha256"] or sha(model / "genai_config.json") != parent["config_sha256"]:
        raise RuntimeError("Frozen graph/config changed")
    command = command_for(prefix, args.context, args.full_validation, args.mode)
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader"], check=True)
    started_ns = time.time_ns()
    subprocess.run(["python3", str(GUARD), "--gpu-index", "7", "--idle-seconds", "3", "--wait-seconds", "15",
                    "--run-seconds", "2400", "--output-prefix", str(prefix), "--", *command], check=True)
    completed_ns = time.time_ns()
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = [json.loads(prefix.with_suffix(suffix).read_text()) for suffix in (".json", ".guard.json")]
    if (guard["command"] != command or raw["genai_native"]["sha256"] != build["runtime_sha256"] or
            raw["plugin_sha256"] != build["provider_sha256"] or raw["profiling"] or raw["comparison"] is not None):
        raise RuntimeError("Wrong natural-run command/binary/profiling state")
    flags = raw["load_experiment_flags"]
    if (flags[FLAG] != str(int(args.mode == "gpu")) or flags["ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE"] != "0" or
            flags["ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE"] != "1" or flags["ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512"] != "0"):
        raise RuntimeError("Wrong natural selector or trace enabled")
    if any(sample["prefill_timing_scope"] != "append_tokens_plus_raw_output_cpu_synchronization"
           for sample in raw["samples"]):
        raise RuntimeError("Unsynchronized prefill measurement")
    workers = verify_workers(raw, "control")
    result = verify_result(raw, guard, dict(provider_sha256=build["provider_sha256"]),
                           parent, references, args.context, args.full_validation)
    log = prefix.with_suffix(".log").read_text()
    if "WEBGPU_GREEDY_ARGMAX rows=" in log or "WEBGPU_GREEDY_MIN_LENGTH rows=" in log:
        raise RuntimeError("Diagnostic trace contaminated natural timing")
    sources = [Path(__file__), BENCHMARK, GUARD, initial_path, primitive_path, mask_path, manifest_path,
               ROOT / "prepare_gpu_greedy_argmax_natural.py", parent_path, controls_path,
               BUILD / "libonnxruntime-genai.so", BUILD / "libonnxruntime_providers_webgpu.so",
               Path(cache["evidence_path"]), *[p for p, _ in prerequisites],
               *[prefix.with_suffix(suffix) for suffix in (".json", ".guard.json", ".log")]]
    proof = dict(status="gpu_greedy_original_prefixes_qualified", mode=args.mode, trial=args.trial,
        context=args.context, full_validation=args.full_validation, result=result, workers=workers,
        runtime_sha256=build["runtime_sha256"], provider_sha256=build["provider_sha256"],
        experiment_flag=FLAG, experiment_flag_value=int(args.mode == "gpu"),
        original_output_prefixes_match=True, zero_major_faults=True,
        same_runtime_and_provider_both_sides=True, control_is_lanes8_with_gpu_greedy_disabled=True,
        prefill_includes_raw_output_cpu_synchronization=True, trace_enabled=False,
        started_ns=started_ns, completed_ns=completed_ns, default_enabled=False, quality_qualified=False,
        source_sha256={str(p): sha(p) for p in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(dict(status=proof["status"], mode=args.mode, context=args.context, result=result)), flush=True)


if __name__ == "__main__":
    main()
