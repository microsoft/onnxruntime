import json
from pathlib import Path
import re
import subprocess
import time

from build_qmoe_node_replay import replace_once, sha
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_argmax_parallel_bitwise import host_path
from run_fc2_private_a_native import BUILD as PROVIDER, FLAG as PRIVATE
from run_gpu_greedy_argmax_ready_standard import BUILD as OLD_BUILD, command_for
from run_gpu_slice_model import verify_sources
from run_gpu_slice_toy_r2 import BUILD, FLAG as SLICE
from run_webgpu_int4_metrics import verify_result, verify_workers


def selectors(variant, context, flag):
    if flag not in (0, 1):
        raise ValueError("Invalid candidate selector")
    if variant == "fc2-private-a-stack-v1" and context in (2048, 8192, 32768):
        return 1, 1, flag
    if variant == "gpu-slice-combined32k-v1" and context == 32768:
        return flag, flag, 0
    raise ValueError("Invalid variant/context")


def prefix_for(variant, context, full, flag, trial, trace):
    return ROOT / f"{variant}-context{context}-{'full' if full else 'retained'}-flag{flag}-trial{trial}-trace{trace}-gpu7"


def run(variant, context, full, flag, trial, trace):
    primary, sliced, private = selectors(variant, context, flag)
    if trace not in (0, 1) or (trace and not full):
        raise ValueError("Routing admission requires full validation")
    prefix = prefix_for(variant, context, full, flag, trial, trace)
    launcher = Path(str(prefix.with_suffix(".json")) + ".launcher.json")
    dump = Path(str(prefix) + "-shaders.txt")
    for path in [launcher, dump, *[prefix.with_suffix(s) for s in (".json", ".log", ".guard.json", ".qualification.json")]]:
        if path.exists():
            raise FileExistsError(path)
    native_path = ROOT / "fc2-private-a-v1-native-outcome.json"
    native = verify_sources(native_path, "fc2_private_a_native_finite_bitwise_and_shader_routing_qualified")
    slice_admission = ROOT / "gpu-slice-v6-r2-model-admission.json"
    verify_sources(slice_admission, "gpu_slice_frozen2k_prefixes_and_routing_admitted")
    manifests = [json.loads((p / "build-manifest.json").read_text()) for p in (BUILD, PROVIDER)]
    for manifest in manifests:
        for p, digest in manifest["source_sha256"].items():
            if sha(host_path(p)) != digest:
                raise RuntimeError(f"Changed build input: {p}")
    runtime_sha, provider_sha = manifests[0]["runtime_sha256"], manifests[1]["provider_sha256"]
    if (sha(BUILD / "libonnxruntime-genai.so") != runtime_sha or
            sha(PROVIDER / "libonnxruntime_providers_webgpu.so") != provider_sha or
            native["provider_sha256"] != provider_sha):
        raise RuntimeError("Wrong qualified binaries")
    slice_provider = json.loads(Path("/home/sayanshaw/repos/qwen38-int4-gpu-slice-copy/build-v1/build-manifest.json").read_text())
    if manifests[1]["parent_provider_sha256"] != slice_provider["provider_sha256"]:
        raise RuntimeError("Private-A provider does not inherit qualified UInt32 Slice")
    prerequisites = [native_path, slice_admission]
    if not trace:
        admission = ROOT / f"{variant}-context{context}-admission.json"
        verify_sources(admission, "decode_stack_frozen_prefixes_and_routing_admitted")
        prerequisites.append(admission)
        if not full:
            full_proof = prefix_for(variant, context, True, flag, trial, 0).with_suffix(".qualification.json")
            verify_sources(full_proof, "decode_stack_original_prefixes_qualified")
            prerequisites.append(full_proof)
    parent_path = ROOT / "qmoe-prefill-rows4-v2-contexts-20261031.qualification.json"
    parent = json.loads(parent_path.read_text())
    controls_path = ROOT / "int4-pending64-standard-contexts.json"
    references = [s for s in json.loads(controls_path.read_text())["samples"] if s["context"] == context]
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    if sha(model / "model.onnx") != parent["graph_sha256"] or sha(model / "genai_config.json") != parent["config_sha256"]:
        raise RuntimeError("Frozen model changed")
    command = command_for(prefix, context, full, "gpu" if primary else "control")
    old = "/workspace/" + str(OLD_BUILD.relative_to("/home/sayanshaw/repos"))
    new = "/workspace/" + str(BUILD.relative_to("/home/sayanshaw/repos"))
    command[-1] = command[-1].replace(old, new)
    command[-1] = replace_once(command[-1], new + "/libonnxruntime_providers_webgpu.so",
        "/workspace/" + str((PROVIDER / "libonnxruntime_providers_webgpu.so").relative_to("/home/sayanshaw/repos")))
    command[-1] = replace_once(command[-1], "benchmark_gpu_greedy_argmax_ready.py", "benchmark_gpu_decode_stack.py")
    command[-1] = replace_once(command[-1], "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE=0",
                              f"ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE={trace}")
    dump_path = "/evidence/webgpu-perf-oct20/" + dump.name if trace else ""
    command[-1] = (f"export {PRIVATE}={private} {SLICE}={sliced} ORT_WEBGPU_SLICE_UINT32_BITCOPY=1 "
        f"ORT_WEBGPU_EP_SHADER_DUMP_FILE={dump_path} ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH=0; " + command[-1])
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
    expected = {SLICE: str(sliced), "ORT_WEBGPU_SLICE_UINT32_BITCOPY": "1",
        "ORT_GENAI_WEBGPU_GREEDY_ARGMAX": str(primary), "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE": str(trace),
        "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH": "0", "ORT_WEBGPU_QMOE_FC2_FINAL_MIX": "1",
        "ORT_WEBGPU_EXPERT_BITCAST_UNPACK": "1", "ORT_WEBGPU_ARGMAX_PARALLEL_BITWISE": "1",
        "ORT_GENAI_CPU_GREEDY_ARGMAX_AVX512": "0"}
    if (guard["command"] != command or raw["genai_native"]["sha256"] != runtime_sha or
            raw["plugin_sha256"] != provider_sha or raw["profiling"] or raw["comparison"] is not None or
            any(flags[k] != v for k, v in expected.items()) or
            json.loads(launcher.read_text()) != dict(status="flags_captured_before_model_load", flags={PRIVATE: str(private)})):
        raise RuntimeError("Wrong binary/flags/command/profiling")
    if any(s["prefill_timing_scope"] != "append_tokens_through_first_token_selection" for s in raw["samples"]):
        raise RuntimeError("Wrong natural prefill boundary")
    workers = verify_workers(raw, "control")
    metrics = verify_result(raw, guard, dict(provider_sha256=provider_sha), parent, references, context, full)
    log = prefix.with_suffix(".log").read_text()
    copies = re.findall(r"^WEBGPU_GPU_SLICE_COPY source_bytes=(\d+) offset_bytes=(\d+) copied_bytes=(\d+)$", log, re.M)
    if trace:
        count = sum(s["output_tokens"] for s in raw["samples"]) if primary else 0
        if (log.count("WEBGPU_GREEDY_ARGMAX rows=") != count or log.count("WEBGPU_GREEDY_MIN_LENGTH rows=") != count or
                "WEBGPU_GREEDY_ARGMAX_CPU_FALLBACK" in log or "WEBGPU_GREEDY_MIN_LENGTH_FUSED" in log):
            raise RuntimeError("Wrong ordinary greedy routing")
        if (not sliced and copies) or (sliced and (len(copies) != len(raw["samples"]) or
                any(int(n) != 248320 * 2 or int(o) + int(n) != int(t) for t, o, n in copies))):
            raise RuntimeError("Wrong last-row GPU Slice routing")
        shader = dump.read_text()
        selected = "=== WebGPU Shader code [QmoeFC2PrivateABitcastF16" in shader
        if selected != bool(private) or (private and "cached_a" not in shader):
            raise RuntimeError("Wrong private-A shader routing")
    elif copies or "WEBGPU_GREEDY_ARGMAX rows=" in log or dump.exists():
        raise RuntimeError("Diagnostics contaminated natural timing")
    sources = [Path(__file__), ROOT / "benchmark_gpu_decode_stack.py", ROOT / "benchmark_gpu_slice.py", GUARD,
        *prerequisites, BUILD / "build-manifest.json", PROVIDER / "build-manifest.json", parent_path, controls_path,
        BUILD / "libonnxruntime-genai.so", PROVIDER / "libonnxruntime_providers_webgpu.so", launcher,
        Path(cache["evidence_path"]), *[prefix.with_suffix(s) for s in (".json", ".guard.json", ".log")]]
    if trace:
        sources.append(dump)
    result = dict(status="decode_stack_original_prefixes_qualified", variant=variant, context=context,
        full_validation=full, flag=flag, trial=trial, trace=trace, runtime_sha256=runtime_sha, provider_sha256=provider_sha,
        selectors=dict(primary=primary, slice=sliced, private_a=private), original_output_prefixes_match=True,
        started_ns=started, completed_ns=ended, workers=workers, slice_transfers=copies, metrics=metrics,
        quality_qualified=False, natural_speedup_claimed=False,
        source_and_evidence_sha256={str(p): sha(p) for p in sources})
    with prefix.with_suffix(".qualification.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "variant", "context", "flag", "trace", "metrics")}), flush=True)
    return result
