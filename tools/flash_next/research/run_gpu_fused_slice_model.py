import json
from pathlib import Path
import re
import subprocess
import time

from build_qmoe_node_replay import replace_once, sha
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_fc2_private_a_native import BUILD as PROVIDER, FLAG as PRIVATE
from run_gpu_greedy_argmax_ready_standard import BUILD as OLD_BUILD, command_for
from run_gpu_slice_model import verify_sources
from run_gpu_slice_toy_r2 import BUILD, FLAG as SLICE
from run_prefill256_model import verify_tokens


VARIANT = "gpu-fused-slice-v1"
FLAG = "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH"
UUID = "GPU-dbecbd7a-fcd4-7a20-89f1-a897e52339b7"
CPUS = list(range(16, 24))


def prefix_for(context, full, flag, trial, trace):
    return ROOT / f"{VARIANT}-context{context}-{'full' if full else 'retained'}-flag{flag}-trial{trial}-trace{trace}-gpu1"


def verify_placement(raw, guard):
    if guard["gpu_index"] != 1 or guard["gpu_uuid"] != UUID:
        raise RuntimeError("Wrong GPU1 adapter")
    masks = raw["thread_cpu_affinity"]
    if (raw["cpu_affinity"] != CPUS or raw["worker_affinity_override"]["target_cpus"] != CPUS or
            not masks or any(mask != CPUS for mask in masks.values())):
        raise RuntimeError("Main thread or workers escaped the matched NUMA0 eight-CPU mask")


def run(context, full, flag, trial, trace):
    if context not in (2048, 8192) or flag not in (0, 1) or trace not in (0, 1) or (trace and not full):
        raise ValueError("Invalid fused-helper experiment scope")
    prefix = prefix_for(context, full, flag, trial, trace)
    launcher = Path(str(prefix.with_suffix(".json")) + ".launcher.json")
    for path in [launcher, *[prefix.with_suffix(s) for s in (".json", ".log", ".guard.json", ".qualification.json")]]:
        if path.exists():
            raise FileExistsError(path)
    native_path = ROOT / "gpu-fused-slice-v1-native-outcome.json"
    native = verify_sources(native_path, "combined_fused_slice_ids_scores_penalties_sampling_exact")
    runtime_sha, provider_sha = native["runtime_sha256"], native["provider_sha256"]
    if sha(BUILD / "libonnxruntime-genai.so") != runtime_sha or sha(PROVIDER / "libonnxruntime_providers_webgpu.so") != provider_sha:
        raise RuntimeError("Changed native-qualified binaries")
    prerequisites = [native_path]
    if not trace:
        admission = ROOT / f"{VARIANT}-context{context}-gpu1-admission.json"
        verify_sources(admission, "fused_slice_frozen_prefixes_and_routing_admitted")
        prerequisites.append(admission)
        if not full:
            path = prefix_for(context, True, flag, trial, 0).with_suffix(".qualification.json")
            verify_sources(path, "fused_slice_original_prefixes_qualified")
            prerequisites.append(path)
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    parent = json.loads(parent_path.read_text())
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [s for s in json.loads(controls_path.read_text())["samples"] if s["context"] == context]
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    if sha(model / "model.onnx") != parent["graph_sha256"] or sha(model / "genai_config.json") != parent["config_sha256"]:
        raise RuntimeError("Frozen graph/config changed")
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    gpu1 = next(line.split() for line in topology.splitlines() if line.startswith("GPU1"))
    if gpu1[-3:] != ["0-47", "0", "N/A"]:
        raise RuntimeError("GPU1 no longer has the verified NUMA0 locality")
    observed_uuid = subprocess.check_output(["nvidia-smi", "--id=1", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if observed_uuid != UUID:
        raise RuntimeError("GPU1 identity changed")
    command = command_for(prefix, context, full, "gpu")
    if command[:3] != ["taskset", "-c", "48-55"]:
        raise RuntimeError("Unexpected original CPU placement")
    command[2] = "16-23"
    command[-1] = replace_once(command[-1], "GPU-0e70e969-bbfc-80c5-d582-88be134632ea", UUID)
    old = "/workspace/" + str(OLD_BUILD.relative_to("/home/sayanshaw/repos"))
    new = "/workspace/" + str(BUILD.relative_to("/home/sayanshaw/repos"))
    command[-1] = command[-1].replace(old, new)
    command[-1] = replace_once(command[-1], new + "/libonnxruntime_providers_webgpu.so",
        "/workspace/" + str((PROVIDER / "libonnxruntime_providers_webgpu.so").relative_to("/home/sayanshaw/repos")))
    command[-1] = replace_once(command[-1], "benchmark_gpu_greedy_argmax_ready.py", "benchmark_gpu_decode_stack.py")
    command[-1] = replace_once(command[-1], "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE=0",
                              f"ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE={trace}")
    command[-1] = (f"export {PRIVATE}=0 {SLICE}=1 ORT_WEBGPU_SLICE_UINT32_BITCOPY=1 "
                   f"{FLAG}={flag} ORT_WEBGPU_EP_SHADER_DUMP_FILE=; " + command[-1])
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["free", "-h"], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader"], check=True)
    started = time.time_ns()
    subprocess.run(["python3", str(GUARD), "--gpu-index", "1", "--idle-seconds", "3",
        "--wait-seconds", "15", "--run-seconds", "2400", "--output-prefix", str(prefix), "--", *command], check=True)
    ended = time.time_ns()
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = [json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json")]
    verify_placement(raw, guard)
    if (raw["status"] != "completed" or guard["status"] != "completed" or guard["exit_code"] != 0 or
            guard["command"] != command or raw["genai_native"]["sha256"] != runtime_sha or
            raw["plugin_sha256"] != provider_sha or raw["graph_sha256"] != parent["graph_sha256"] or
            raw["config_sha256"] != parent["config_sha256"] or raw["provider"] != "webgpu" or raw["chunk"] != 1024 or
            raw["webgpu_max_pending_dispatches_override"] != 64 or
            raw["webgpu_validation_mode_override"] != ("full" if full else None) or
            raw["profiling"] or raw["profiling_scope"] is not None or raw["comparison"] is not None or
            raw["profile_decode_after_tokens"] is not None):
        raise RuntimeError("Wrong natural model execution scope/metadata")
    expected = {key: "1" for key in ("ORT_WEBGPU_QMOE_PREFILL256", "ORT_WEBGPU_QMOE_EXTENDED_PREFILL",
        "ORT_WEBGPU_QSA_INDEXER_STACK", "ORT_WEBGPU_DENSE_BITCAST_UNPACK", "ORT_WEBGPU_DENSE_TILE4",
        "ORT_WEBGPU_HC_UP_PARALLEL2", "ORT_WEBGPU_QSA_SCORE16", "ORT_WEBGPU_QMOE_PREFILL512",
        "ORT_WEBGPU_QMOE_PREFILL1024", "ORT_WEBGPU_QMOE_GROUPED_PREFILL", "ORT_WEBGPU_DSA_QUERY_CACHE",
        "ORT_WEBGPU_DENSE_BLOCK32", "ORT_WEBGPU_QSA_PREFILL_SCORE16", "ORT_WEBGPU_QMOE_FC2_FINAL_MIX",
        "ORT_GENAI_WEBGPU_GREEDY_ARGMAX", "ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE", SLICE, "ORT_WEBGPU_SLICE_UINT32_BITCOPY")}
    expected.update({key: "0" for key in ("ORT_WEBGPU_QSA_CONCAT_VECTOR", "ORT_WEBGPU_QSA_PACKED_SCORE",
        "ORT_WEBGPU_QMOE_FC2_SUBGROUP_FOLD", "ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512")})
    expected.update({FLAG: str(flag), "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE": str(trace)})
    if (any(raw["load_experiment_flags"][k] != v for k, v in expected.items()) or
            json.loads(launcher.read_text()) != dict(status="flags_captured_before_model_load", flags={PRIVATE: "0"}) or
            any(s["prefill_timing_scope"] != "append_tokens_through_first_token_selection" for s in raw["samples"])):
        raise RuntimeError("Wrong inherited flags, private FC2 enabled, or wrong GPU-ready boundary")
    metrics = verify_tokens(raw, references, context, 256)
    log = prefix.with_suffix(".log").read_text()
    copies = re.findall(r"^WEBGPU_GPU_SLICE_COPY source_bytes=(\d+) offset_bytes=(\d+) copied_bytes=(\d+)$", log, re.M)
    if trace:
        count = sum(s["output_tokens"] for s in raw["samples"])
        if (log.count("WEBGPU_GREEDY_ARGMAX rows=") != count or log.count("WEBGPU_GREEDY_MIN_LENGTH rows=") != count or
                log.count("WEBGPU_GREEDY_MIN_LENGTH_FUSED") != count * flag or
                log.count("WEBGPU_GREEDY_ARGMAX_CACHED") != count * flag or
                "WEBGPU_GREEDY_ARGMAX_CPU_FALLBACK" in log or len(copies) != len(raw["samples"]) or
                any(int(n) != 248320 * 2 or int(o) + int(n) != int(t) for t, o, n in copies)):
            raise RuntimeError("Incomplete fused/ordinary greedy or Slice routing")
    elif copies or "WEBGPU_GREEDY_" in log:
        raise RuntimeError("Diagnostic trace contaminated natural timing")
    sources = [Path(__file__), ROOT / "benchmark_gpu_decode_stack.py", ROOT / "benchmark_gpu_slice.py", GUARD,
        ROOT / "run_prefill256_model.py", ROOT / "run_gpu_greedy_argmax_ready_standard.py", *prerequisites,
        BUILD / "libonnxruntime-genai.so", PROVIDER / "libonnxruntime_providers_webgpu.so", parent_path, controls_path,
        launcher, Path(cache["evidence_path"]), *[prefix.with_suffix(s) for s in (".json", ".guard.json", ".log")]]
    result = dict(status="fused_slice_original_prefixes_qualified", context=context, full_validation=full,
        flag=flag, trial=trial, trace=trace, gpu_index=1, gpu_uuid=UUID, cpu_affinity=CPUS,
        runtime_sha256=runtime_sha, provider_sha256=provider_sha, started_ns=started, completed_ns=ended,
        original_output_prefixes_match=True, slice_transfers=copies, metrics=metrics,
        quality_qualified=False, natural_speedup_claimed=False, cross_gpu_absolute_comparison_qualified=False,
        source_and_evidence_sha256={str(p): sha(p) for p in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "context", "flag", "trace", "metrics")}), flush=True)
    return result
