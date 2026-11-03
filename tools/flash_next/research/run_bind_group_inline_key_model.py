import json
from pathlib import Path
import re
import statistics
import subprocess
import time

from build_qmoe_node_replay import replace_once, sha
from model_cache_preflight import prepare_model_cache, snapshot_model_cache
from prepare_webgpu_int4_metrics import GUARD, ROOT
from run_bind_group_inline_key_native import BUILD, FLAG
from run_gpu_decode_stack_pipeline import compare
from run_gpu_fused_slice_model import verify_placement
from run_gpu_slice_model import verify_sources
from run_gpu_slice_pipeline import percentile
from run_prefill256_model import verify_tokens


VARIANT = "bind-group-inline-key-v1"
TEMPLATE = "gpu-fused-slice-v1-context2048-full-flag0-trial0-trace0-gpu1"


def prefix_for(context, full, flag, trial, trace):
    return ROOT / f"{VARIANT}-context{context}-{'full' if full else 'retained'}-flag{flag}-trial{trial}-trace{trace}-gpu1"


def command_for(prefix, context, full, flag, trace):
    command = json.loads((ROOT / (TEMPLATE + ".guard.json")).read_text())["command"].copy()
    if command[:5] != ["taskset", "-c", "16-23", "bash", "-lc"] or command[-1].count(TEMPLATE) != 3:
        raise RuntimeError("Unexpected baseline launch template")
    shell = command[-1].replace(TEMPLATE, prefix.name)
    shell = replace_once(shell, "benchmark_gpu_decode_stack.py", "benchmark_bind_group_inline_key.py")
    shell = replace_once(shell, "--contexts 2048", f"--contexts {context}")
    shell = replace_once(shell, "ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE=0",
                         f"ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE={trace}")
    shell = replace_once(shell,
        "/workspace/qwen38-int4-fc2-private-a/build-v1/libonnxruntime_providers_webgpu.so",
        "/workspace/qwen38-int4-bind-group-inline-key/build-v1/libonnxruntime_providers_webgpu.so")
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


def run(context, full, flag, trial, trace):
    if context not in (2048, 8192) or flag not in (0, 1) or trace not in (0, 1) or (trace and not full):
        raise ValueError("Invalid context-cache experiment scope")
    prefix = prefix_for(context, full, flag, trial, trace)
    for path in (prefix.with_suffix(s) for s in (".json", ".log", ".guard.json", ".qualification.json")):
        if path.exists():
            raise FileExistsError(path)
    if Path(str(prefix) + ".json.launcher.json").exists():
        raise FileExistsError(str(prefix) + ".json.launcher.json")
    native_path = ROOT / "bind-group-inline-key-v1-native-outcome.json"
    native = verify_sources(native_path, "bind_group_inline_key_native_finite_bitwise_qualified")
    if sha(BUILD / "libonnxruntime_providers_webgpu.so") != native["provider_sha256"]:
        raise RuntimeError("Changed native-qualified provider")
    verify_sources(ROOT / (TEMPLATE + ".qualification.json"), "fused_slice_original_prefixes_qualified")
    if not trace:
        verify_sources(ROOT / f"{VARIANT}-context{context}-gpu1-admission.json", "bind_group_inline_key_frozen_admitted")
        if not full:
            verify_sources(prefix_for(context, True, flag, trial, 0).with_suffix(".qualification.json"),
                           "bind_group_inline_key_original_prefixes_qualified")
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
            guard["command"] != command or raw["plugin_sha256"] != native["provider_sha256"] or
            raw["genai_native"] != baseline["genai_native"] or raw["ort_native"] != baseline["ort_native"] or
            raw["graph_sha256"] != baseline["graph_sha256"] or raw["config_sha256"] != baseline["config_sha256"] or
            raw["provider"] != "webgpu" or raw["chunk"] != 1024 or
            raw["webgpu_max_pending_dispatches_override"] != 64 or
            raw["webgpu_validation_mode_override"] != ("full" if full else None) or
            raw["profiling"] or raw["profiling_scope"] is not None or raw["comparison"] is not None or
            raw["profile_decode_after_tokens"] is not None):
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
    verify_tokens(raw, [s for s in references if s["context"] == context], context, 256)
    log = prefix.with_suffix(".log").read_text()
    verify_routing(raw, log, trace)
    sources = (Path(__file__), ROOT / "benchmark_bind_group_inline_key.py", ROOT / "benchmark_gpu_slice.py",
        ROOT / "run_prefill256_model.py", ROOT / "run_gpu_fused_slice_model.py",
        ROOT / (TEMPLATE + ".qualification.json"), ROOT / (TEMPLATE + ".json"),
        ROOT / (TEMPLATE + ".guard.json"), BUILD / "libonnxruntime_providers_webgpu.so",
        GUARD, native_path, launcher, prefix.with_suffix(".json"), prefix.with_suffix(".guard.json"),
        prefix.with_suffix(".log"))
    proof = dict(status="bind_group_inline_key_original_prefixes_qualified", context=context,
        full_validation=full, flag=flag, trial=trial, trace=trace, gpu_index=1,
        started_ns=started, completed_ns=ended, quality_qualified=False,
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
            proof = verify_sources(path, "bind_group_inline_key_original_prefixes_qualified")
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
    result = dict(status="bind_group_inline_key_gain_both_modes" if supported else "bind_group_inline_key_no_supported_gain",
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
            sources = [Path(__file__), *[prefix_for(context, True, flag, 0, 1).with_suffix(".qualification.json") for flag in (0, 1)]]
            admission = dict(status="bind_group_inline_key_frozen_admitted", context=context, gpu_index=1,
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
                if result["status"] != "bind_group_inline_key_gain_both_modes":
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
