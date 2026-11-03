import collections
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

from build_qmoe_node_replay import replace_once, sha
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from prepare_webgpu_int4_metrics import GUARD, ROOT
from dense_tile_parallel3_long_k_transform import FLAG, NAME
from run_dense_long_k_native_v3 import BUILD
from run_argmax_parallel_bitwise import host_path
from run_dawn_scope_qualification_v1 import PARENT, container
from run_gpu_decode_stack_pipeline import compare
from run_gpu_fused_slice_model import verify_placement
from run_gpu_slice_model import verify_sources
from run_gpu_slice_pipeline import percentile
from run_prefill256_model import verify_tokens


VARIANT = "dense-tile-parallel3-long-k-model-v1"
TEMPLATE = "gpu-fused-slice-v1-context2048-full-flag0-trial0-trace0-gpu1"


def prefix_for(context, full, flag, trial, trace):
    return ROOT / f"{VARIANT}-context{context}-{'full' if full else 'retained'}-flag{flag}-trial{trial}-trace{trace}-gpu1"


def command_for(prefix, context, full, flag, trace):
    command = json.loads((ROOT / (TEMPLATE + ".guard.json")).read_text())["command"].copy()
    if command[:5] != ["taskset", "-c", "16-23", "bash", "-lc"] or command[-1].count(TEMPLATE) != 3:
        raise RuntimeError("Unexpected baseline launch template")
    shell = command[-1].replace(TEMPLATE, prefix.name)
    shell = replace_once(shell, "benchmark_gpu_decode_stack.py", "benchmark_dense_long_k.py")
    shell = replace_once(shell, "--contexts 2048", f"--contexts {context}")
    shell = replace_once(shell, "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE=0",
                         f"ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE={trace}")
    provider = (BUILD if flag else PARENT) / "libonnxruntime_providers_webgpu.so"
    shell = replace_once(shell,
        "/workspace/qwen38-int4-fc2-private-a/build-v1/libonnxruntime_providers_webgpu.so", container(provider))
    if trace:
        shell = replace_once(shell, "--tokens 256 --repetitions 3",
                             f"--tokens 36 --repetitions 1 --profile-decode-after 35 "
                             f"--profile-prefix /evidence/webgpu-perf-oct20/{prefix.name}-profile")
    if not full:
        shell = replace_once(shell, " --webgpu-validation-mode full", "")
    command[-1] = f"export {FLAG}={flag}; " + shell
    return command


def verify_routing(raw, log, trace):
    calls = sum(s["output_tokens"] for s in raw["samples"])
    copies = re.findall(r"^WEBGPU_GPU_SLICE_COPY source_bytes=(\d+) offset_bytes=(\d+) copied_bytes=(\d+)$", log, re.M)
    if trace:
        if (len(re.findall(r"^WEBGPU_GREEDY_ARGMAX rows=", log, re.M)) != calls or
                len(re.findall(r"^WEBGPU_GREEDY_MIN_LENGTH rows=", log, re.M)) != calls or
                "WEBGPU_GREEDY_MIN_LENGTH_FUSED" in log or "WEBGPU_GREEDY_ARGMAX_CACHED" in log or
                "WEBGPU_GREEDY_ARGMAX_CPU_FALLBACK" in log or len(copies) != len(raw["samples"])):
            raise RuntimeError("Missing ordinary-greedy/Slice routing or an unsupported search route")
        if any(tuple(map(int, row)) != (508559360, 508062720, 496640) for row in copies):
            raise RuntimeError("Wrong final-chunk Slice dimensions")
    elif copies or "WEBGPU_GREEDY_" in log:
        raise RuntimeError("Trace unexpectedly enabled during natural timing")



def model_programs(events, flag):
    if flag not in (0, 1):
        raise ValueError("Invalid long-K model profile scope")
    kernels = [event for event in events if event.get("cat") == "Kernel"]
    private = [event for event in kernels if event.get("name", "").rsplit("&", 1)[-1] == NAME]
    if len(private) != (48 if flag else 0):
        raise RuntimeError("Wrong actual long-K model dispatch count")
    for event in private:
        if (event.get("pid") != -1 or event.get("tid") != -1 or event.get("ph") != "X"
                or not math.isfinite(event.get("dur", 0)) or event.get("dur", 0) <= 0
                or ":384,1,1:" not in event.get("args", {}).get("cache_key", "")):
            raise RuntimeError("Invalid actual long-K GPU event/workgroup")
    return dict(collections.Counter(
        "DenseColumns8DoubleBuffer" if event.get("name", "").rsplit("&", 1)[-1] == NAME
        else event.get("name", "").rsplit("&", 1)[-1] for event in kernels))


def verify_model_profiles(raw, flag):
    profiles, counts = [], []
    for sample in raw["samples"]:
        files = sample.get("decode_profile_files", [])
        if len(files) != 1 or sample["profiled_decode_tokens"] != 1 or sample["profile_after_tokens"] != 35:
            raise RuntimeError("Expected exactly one late decode GPU profile per request")
        profile = host_path(files[0])
        counts.append(model_programs(json.loads(profile.read_text()), flag))
        profiles.append(profile)
    return profiles, counts


def run(context, full, flag, trial, trace):
    if context not in (2048, 8192) or flag not in (0, 1) or trace not in (0, 1) or (trace and not full):
        raise ValueError("Invalid context-cache experiment scope")
    prefix = prefix_for(context, full, flag, trial, trace)
    for path in (prefix.with_suffix(s) for s in (".json", ".log", ".guard.json", ".qualification.json")):
        if path.exists():
            raise FileExistsError(path)
    if Path(str(prefix) + ".json.launcher.json").exists():
        raise FileExistsError(str(prefix) + ".json.launcher.json")
    native_path = ROOT / "dense-tile-parallel3-long-k-native-v3-outcome.json"
    native = verify_sources(native_path, "dense_long_k_fixed_process_native_golden_qualified")
    build = json.loads((BUILD / "build-manifest.json").read_text())
    provider = (BUILD if flag else PARENT) / "libonnxruntime_providers_webgpu.so"
    expected_provider_sha = native["provider_sha256"] if flag else build["baseline_provider_sha256"]
    if (not native["native_model_admission_eligible"] or sha(provider) != expected_provider_sha
            or sha(BUILD / "libonnxruntime_providers_webgpu.so") != native["provider_sha256"]):
        raise RuntimeError("Changed native-qualified or independent original provider")
    verify_sources(ROOT / (TEMPLATE + ".qualification.json"), "fused_slice_original_prefixes_qualified")
    if not trace:
        verify_sources(ROOT / f"{VARIANT}-context{context}-gpu1-admission.json", "dense_long_k_frozen_admitted")
        if not full:
            verify_sources(prefix_for(context, True, flag, trial, 0).with_suffix(".qualification.json"),
                           "dense_long_k_original_prefixes_qualified")
    baseline = json.loads((ROOT / (TEMPLATE + ".json")).read_text())
    model = Path("/home/sayanshaw/repos/qwen38-int4-mlp-input-projections/model")
    for name, key in (("model.onnx", "graph_sha256"), ("genai_config.json", "config_sha256")):
        if sha(model / name) != baseline[key]:
            raise RuntimeError("Frozen model changed")
    command = command_for(prefix, context, full, flag, trace)
    cache = prepare_model_cache(model, prefix)
    subprocess.run(["free", "-h"], check=True)
    subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv"], check=True)
    started = time.time_ns()
    subprocess.run(["python3", str(GUARD), "--gpu-index", "1", "--idle-seconds", "3",
        "--wait-seconds", "15", "--run-seconds", "2400", "--output-prefix", str(prefix), "--", *command], check=True)
    ended = time.time_ns()
    cache["after_run"] = snapshot_model_cache(model)
    Path(cache["evidence_path"]).write_text(json.dumps(cache, indent=2) + "\n")
    raw, guard = [json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json")]
    verify_placement(raw, guard)
    if (raw["status"] != "completed" or guard["status"] != "completed" or guard["exit_code"] != 0 or
            guard["command"] != command or raw["plugin_sha256"] != expected_provider_sha or
            raw["genai_native"] != baseline["genai_native"] or raw["ort_native"] != baseline["ort_native"] or
            raw["graph_sha256"] != baseline["graph_sha256"] or raw["config_sha256"] != baseline["config_sha256"] or
            raw["provider"] != "webgpu" or raw["chunk"] != 1024 or
            raw["webgpu_max_pending_dispatches_override"] != 64 or
            raw["webgpu_validation_mode_override"] != ("full" if full else None) or
            raw["profiling"] != bool(trace) or raw["profiling_scope"] != ("decode_only" if trace else None) or
            raw["comparison"] is not None or raw["profile_decode_after_tokens"] != (35 if trace else None)):
        raise RuntimeError("Wrong natural model execution scope")
    expected = dict(baseline["load_experiment_flags"])
    expected["ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE"] = str(trace)
    if raw["load_experiment_flags"] != expected:
        raise RuntimeError("Unexpected inherited execution flags")
    launcher = Path(str(prefix) + ".json.launcher.json")
    flags = json.loads(launcher.read_text())
    if flags != dict(status="flags_captured_before_model_load", flags={
            FLAG: str(flag), "ORT_WEBGPU_QMOE_FC2_PRIVATE_A": "0",
            "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_FUSED_MIN_LENGTH": "0",
            "ORT_GENAI_WEBGPU_GPU_SLICE_COPY": "1",
            "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES": "8192", "ORT_WEBGPU_DYNAMIC_UNIFORMS": "1"}):
        raise RuntimeError("Wrong process-static settings at model load")
    if any(sample["prefill_timing_scope"] != "append_tokens_through_first_token_selection" for sample in raw["samples"]):
        raise RuntimeError("Wrong prefill-ready boundary")
    references = json.loads((ROOT / "int4-pending64-standard-contexts.json").read_text())["samples"]
    verify_tokens(raw, [s for s in references if s["context"] == context], context, 36 if trace else 256)
    log = prefix.with_suffix(".log").read_text()
    verify_routing(raw, log, trace)
    profiles, program_counts = verify_model_profiles(raw, flag) if trace else ([], [])
    sources = (Path(__file__), ROOT / "benchmark_dense_long_k.py", ROOT / "benchmark_gpu_slice.py",
        ROOT / "run_prefill256_model.py", ROOT / "run_gpu_fused_slice_model.py",
        ROOT / (TEMPLATE + ".qualification.json"), ROOT / (TEMPLATE + ".json"),
        ROOT / (TEMPLATE + ".guard.json"), BUILD / "libonnxruntime_providers_webgpu.so",
        GUARD, native_path, launcher, provider, BUILD / "build-manifest.json",
        ROOT / "prepare_dense_long_k_model_v1.py", *profiles, prefix.with_suffix(".json"), prefix.with_suffix(".guard.json"),
        prefix.with_suffix(".log"))
    proof = dict(status="dense_long_k_original_prefixes_qualified", context=context,
        full_validation=full, flag=flag, trial=trial, trace=trace, gpu_index=1,
        started_ns=started, completed_ns=ended, quality_qualified=False,
        original_provider_control=flag == 0, provider_sha256=expected_provider_sha,
        normalized_program_counts=program_counts,
        source_and_evidence_sha256={str(path): sha(path) for path in sources})
    prefix.with_suffix(".qualification.json").write_text(json.dumps(proof, indent=2) + "\n")


def screen(context, repeat):
    baseline = int(repeat)
    rows, sources = [], [Path(__file__), ROOT / "run_gpu_decode_stack_pipeline.py"]
    for full in (True, False):
        groups, times = [], []
        for flag, trial in ((0, baseline), (1, baseline), (0, baseline + 1)):
            if not (repeat and flag == 0 and trial == baseline):
                run(context, full, flag, trial, 0)
            prefix = prefix_for(context, full, flag, trial, 0)
            path = prefix.with_suffix(".qualification.json")
            proof = verify_sources(path, "dense_long_k_original_prefixes_qualified")
            raw, guard = [json.loads(prefix.with_suffix(s).read_text()) for s in (".json", ".guard.json")]
            samples = [s for s in raw["samples"] if not s["warmup"]]
            if (proof["context"] != context or proof["full_validation"] != full or proof["flag"] != flag or
                    proof["trial"] != trial or proof["trace"] != 0 or proof["gpu_index"] != 1 or
                    guard["status"] != "completed" or len(samples) != 3 or len(raw["samples"]) != 4):
                raise RuntimeError("Wrong bracketed context-cache samples")
            groups.append(samples)
            times.append((proof["started_ns"], proof["completed_ns"]))
            sources.append(path)
            if flag:
                candidate = dict(full_validation=full, decode_tps=3 * 255 / sum(s["decode_seconds"] for s in samples),
                    median_decode_tps=statistics.median(s["decode_tps"] for s in samples),
                    prefill_tps=sum(s["input_tokens"] for s in samples) / sum(s["prefill_seconds"] for s in samples),
                    ttft_p50_seconds=percentile([s["ttft_seconds"] for s in samples], .5),
                    ttft_p95_seconds=percentile([s["ttft_seconds"] for s in samples], .95),
                    sampled_peak_gpu_gib=guard["peak_gpu_used_bytes"] / 2**30)
        if not times[0][1] <= times[1][0] <= times[1][1] <= times[2][0]:
            raise RuntimeError("Samples are not time-bracketed")
        rows.append(dict(candidate=candidate, comparison=compare(groups[1], groups[0], groups[2])))
    supported = all(row["comparison"]["supported"] for row in rows)
    result = dict(status="dense_long_k_gain_both_modes" if supported else "dense_long_k_no_supported_gain",
        context=context, repeat=repeat, rows=rows, gpu_index=1, reused_initial_ending_controls=repeat,
        quality_qualified=False, default_enabled=False, cross_gpu_absolute_comparison_qualified=False,
        prefill_scope="append_tokens_through_first_token_selection",
        source_and_evidence_sha256={str(path): sha(path) for path in sources})
    path = ROOT / f"{VARIANT}-context{context}-{'repeat-' if repeat else ''}gpu1-screen.json"
    with path.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return path, result


if __name__ == "__main__":
    state_path = ROOT / f"{VARIANT}-gpu1-pipeline.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    state = dict(status="starting", reports=[], goal_reached=False, started_ns=time.time_ns())

    def persist():
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    persist()
    try:
        for context in (2048, 8192):
            state.update(status="frozen_admission", context=context)
            persist()
            for flag in (0, 1):
                run(context, True, flag, 0, 1)
            routing = [json.loads(prefix_for(context, True, flag, 0, 1).with_suffix(
                ".qualification.json").read_text()) for flag in (0, 1)]
            if routing[0]["normalized_program_counts"] != routing[1]["normalized_program_counts"]:
                raise RuntimeError("Unchanged model GPU program counts differ after exact long-K substitution")
            sources = [Path(__file__), *[prefix_for(context, True, flag, 0, 1).with_suffix(".qualification.json") for flag in (0, 1)]]
            admission = dict(status="dense_long_k_frozen_admitted", context=context, gpu_index=1,
                quality_qualified=False, performance_qualified=False,
                source_and_evidence_sha256={str(path): sha(path) for path in sources})
            path = ROOT / f"{VARIANT}-context{context}-gpu1-admission.json"
            with path.open("x") as stream:
                stream.write(json.dumps(admission, indent=2) + "\n")
            for repeat in (False, True):
                state.update(status="natural_decode_screen", repeat=repeat)
                persist()
                path, result = screen(context, repeat)
                state["reports"].append(dict(path=str(path), **{key: result[key] for key in ("context", "repeat", "status", "rows")}))
                if result["status"] != "dense_long_k_gain_both_modes":
                    state["status"] = "held_no_unchanged_retry"
                    persist()
                    raise SystemExit(0)
                persist()
        state["status"] = "bounded_2k_8k_complete_32k_requires_supported_control"
        persist()
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        persist()
        raise
