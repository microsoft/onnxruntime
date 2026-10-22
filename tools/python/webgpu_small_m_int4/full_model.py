# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Paired genuine-Qwen target replay and DFlash2 Engine benchmarks."""

import argparse
import gc
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from reference import errors

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = Path("/workspace/onnxruntime-genai/build/qwen38-narrow-drafters-20261020")
ROUTING = "ORT_WEBGPU_EXPERIMENTAL_SMALL_M_SIMT"
GPU_PROFILE = "ORT_WEBGPU_DIAGNOSTIC_GPU_PROFILE"
MODES = {"baseline": ("w7", 0), "w1": ("true-w1", 1), "w3": ("true-w3", 3)}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapped_binaries(plugin):
    paths = {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if ".so" in line
        and line.split()[-1].startswith("/")
        and any(name in line for name in ("libonnxruntime", "onnxruntime_pybind11", "onnxruntime_genai.cpython"))
    }
    if plugin.resolve() not in paths:
        raise AssertionError("The requested benchmark plugin was not loaded.")
    return {str(path): file_hash(path) for path in sorted(paths)}


def load_runtime(args):
    if GPU_PROFILE in os.environ:
        raise ValueError("Set GPU profiling through this harness, not the caller environment.")
    for name in ("ORT_WEBGPU_SMALL_M_DIAGNOSTIC", "ORT_WEBGPU_FORCE_CPU_NODE_NAMES"):
        if name in os.environ:
            raise ValueError(f"Conflicting diagnostic environment: {name}")
    os.environ[ROUTING] = "0"
    sys.path[:0] = [
        str(args.runtime_root / "ort-core/Release"),
        str(args.runtime_root / "genai-build/Release/wheel"),
    ]
    ort, og = importlib.import_module("onnxruntime"), importlib.import_module("onnxruntime_genai")
    for module, directory in ((ort, "ort-core"), (og, "genai-build")):
        if not Path(module.__file__).resolve().is_relative_to(args.runtime_root / directory):
            raise AssertionError(f"Unexpected runtime package: {module.__file__}")
    ort.register_execution_provider_library("small_m_simt", str(args.plugin))
    og.register_execution_provider_library("small_m_simt", str(args.plugin))
    return ort, og


def metadata(args, ort, og):
    return {
        "command": [sys.executable, *sys.argv],
        "ORT_source_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "ORT_source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(ROOT), "diff", "HEAD"])
        ).hexdigest(),
        "ORT_build_info": ort.get_build_info(),
        "ORT_package": ort.__file__,
        "GenAI_package": og.__file__,
        "GenAI_build_commit": og.__commit__,
        "plugin": str(args.plugin),
        "plugin_sha256": file_hash(args.plugin),
        "runtime_root": str(args.runtime_root),
        "kind": args.kind,
        "routing_variable": ROUTING,
        "routing": "Fixed padded BN8/BK512; exact MLP shapes only; all other selectors unchanged.",
    }


def profile_summary(path):
    kernels = [event for line in path.read_text().splitlines() for event in json.loads(line)["kernels"]]
    programs, routes = Counter(), Counter()
    role_us, role_programs = defaultdict(float), defaultdict(Counter)
    projection = 0
    for event in kernels:
        node, op, program = event["name"].split("&")
        programs[program] += 1
        if program == "DiagnosticSmallMSimt":
            b = re.search(r"inputs\[1\] = \{([^}]+)\}", event["shapes"])
            y = re.search(r"outputs\[0\] = \{([^}]+)\}", event["shapes"])
            if b is None or y is None:
                raise AssertionError("Missing actual SIMT input/output shapes.")
            bshape = [int(d) for d in b[1].split(",")]
            yshape = [int(d) for d in y[1].split(",")]
            m, n, k = int(np.prod(yshape[:-1])), bshape[0], bshape[1] * 32
            if m not in (2, 4, 8) or (k, n) not in ((5120, 17408), (17408, 5120)):
                raise AssertionError(f"Ineligible actual SIMT route: {(m, k, n)}")
            if f"[{m}|8|512|4|1|" not in event["cache_key"]:
                raise AssertionError("The executed SIMT shader is not fixed padded BN8/BK512.")
            routes[f"M{m}_K{k}_N{n}"] += 1
        is_projection = program in {
            "MatMulNBits",
            "DiagnosticSmallMSimt",
            "SubgroupMatrixMatMulNBits",
            "DP4AMatMulNBits",
            "MatMulNBitsWideTile",
        }
        if op == "MatMulNBitsMlp" and (is_projection or program == "SubgroupMatrixMatMulLayout"):
            role = "gate" if projection % 2 == 0 else "up"
            role_us[role] += event["dur"]
            role_programs[role][program] += 1
            if is_projection:
                projection += 1
        else:
            for role in ("gate", "up", "down"):
                if f"/mlp/{role}_proj/" in node:
                    role_us[role] += event["dur"]
                    role_programs[role][program] += 1
                    break
    return {
        "gpu_ms": sum(event["dur"] for event in kernels) / 1000,
        "dispatches": len(kernels),
        "programs": dict(programs),
        "simt_routes": dict(routes),
        "mlp_gpu_ms": {role: duration / 1000 for role, duration in role_us.items()},
        "mlp_programs": {role: dict(counts) for role, counts in role_programs.items()},
    }


def validate_assignment(paths):
    gpu, cpu = Counter(), Counter()
    for path in paths:
        for event in json.loads(path.read_text()):
            args = event.get("args", {})
            if args.get("provider") == "WebGpuExecutionProvider" and args.get("op_name"):
                gpu[args["op_name"]] += 1
            elif args.get("provider") == "CPUExecutionProvider" and args.get("op_name"):
                cpu[args["op_name"]] += 1
    heavy = {
        "MatMulNBits",
        "MatMulNBitsMlp",
        "PagedAttention",
        "GatedDeltaNet",
        "Conv",
        "ConvStateReplay",
        "GatedDeltaNetStateReplay",
    }
    if not paths or not gpu or heavy & cpu.keys():
        raise AssertionError(f"Missing GPU execution or heavy CPU fallback: GPU={gpu}, CPU={cpu}")
    return {"profiles": [str(path) for path in paths], "gpu_ops": dict(gpu), "cpu_ops": dict(cpu)}


def controlled(args, ort, og):
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "WebGpuExecutionProvider"]
    device = next(d for d in devices if d.device.metadata.get("pci_bus_id") == args.pci_bus)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.add_provider_for_devices(
        [device], {"dawnBackendType": "Vulkan", "validationMode": "basic", "enableGraphCapture": "0"}
    )
    if args.kind == "assignment":
        options.enable_profiling = True
        options.profile_file_prefix = str(args.output / "nodes")
    model = args.runtime_root.parent / "pr2606-rebuilt-model-20261014/model.onnx"
    session = ort.InferenceSession(str(model), options, providers=[])
    dimensions = {"batch_size": 1, "num_blocks": 64, "block_size": 256}
    dtypes = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    state_meta = [m for m in session.get_inputs() if m.name.startswith("past.")]
    state_meta += [m for m in session.get_inputs() if m.name.startswith("past_key_values.")]
    cpu_inputs = {"cumulative_sequence_lengths", "past_sequence_lengths", "attention_metadata", "state_update_active"}
    result = {
        **metadata(args, ort, og),
        "model": str(model),
        "model_sha256": file_hash(model),
        "conditions": [],
        "device": device.device.metadata,
        "cpu_fallback_policy": "Existing metadata nodes only; separate assignment profile checks heavy operations.",
    }

    def alloc(shape, dtype):
        return session.create_ortvalue_from_shape_and_type(shape, dtype, "webgpu")

    def upload(array):
        value = alloc(array.shape, array.dtype)
        ort.copy_tensors([ort.OrtValue.ortvalue_from_numpy(array)], [value])
        return value

    def read(values):
        arrays = [np.empty(v.shape(), dtypes[v.data_type()]) for v in values]
        ort.copy_tensors(values, [ort.OrtValue.ortvalue_from_numpy(array) for array in arrays])
        return arrays

    def digest(arrays):
        value = hashlib.sha256()
        for array in arrays:
            value.update(array.tobytes())
        return value.hexdigest()

    states = {
        meta.name: alloc([dimensions[d] if isinstance(d, str) else d for d in meta.shape], dtypes[meta.type])
        for meta in state_meta
    }
    state_names, state_values = list(states), list(states.values())
    state_out = {name.replace("past_key_values.", "present.").replace("past.", "present."): name for name in states}
    workloads = json.loads((args.runtime_root / "workloads.json").read_text())
    baseline = json.loads((args.runtime_root / "timing-baseline-w1/result.json").read_text())
    for length, workload_name in ((100, "short"), (1500, "long")):
        workload = next(w for w in workloads if w["name"] == workload_name)
        trial = next(
            t for t in baseline["trials"] if t["workload"] == workload_name and t["configuration"] == "baseline"
        )
        candidates = (workload["prompt_ids"] + trial["output_ids"])[length : length + 8]
        if len(candidates) != 8:
            raise AssertionError("Not enough identical frozen-context candidate tokens.")
        with np.load(args.state_dir / f"state-context{length}.npz") as saved:
            arrays = [saved[name] for name in state_names]
        snapshot_hash = digest(arrays)
        snapshots = [upload(array) for array in arrays]
        groups, conditions = {}, {}
        for m in (2, 4, 8):
            feeds = {
                "input_ids": np.asarray(candidates[:m], np.int64),
                "position_ids": np.tile(np.arange(length, length + m, dtype=np.int64), (3, 1)),
                "cumulative_sequence_lengths": np.asarray([0, m], np.int32),
                "past_sequence_lengths": np.asarray([length], np.int32),
                "block_table": np.arange(64, dtype=np.int32).reshape(1, 64),
                "attention_metadata": np.asarray([m, length + m, length + m], np.int32),
                "state_update_capture_count": np.asarray([m - 1], np.int32),
                "state_update_active": np.asarray([1], np.int32),
            }
            io, keep, outputs = session.io_binding(), {}, {}
            for name, value in states.items():
                io.bind_ortvalue_input(name, value)
            for name, array in feeds.items():
                keep[name] = ort.OrtValue.ortvalue_from_numpy(array) if name in cpu_inputs else upload(array)
                io.bind_ortvalue_input(name, keep[name])
            for meta in session.get_outputs():
                value = (
                    states[state_out[meta.name]]
                    if meta.name in state_out
                    else alloc(
                        [dimensions.get(d, m) if isinstance(d, str) else d for d in meta.shape], dtypes[meta.type]
                    )
                )
                io.bind_ortvalue_output(meta.name, value)
                outputs[meta.name] = value
            groups[m] = io, keep, outputs
            for route in (0, 1):
                record = {
                    "context": length,
                    "M": m,
                    "routing": route,
                    "tokens": candidates[:m],
                    "positions": list(range(length, length + m)),
                    "state_sha256": snapshot_hash,
                    "state_tensors": len(states),
                    "warmup_host_ms": [],
                    "calls": [],
                }
                conditions[(m, route)] = record
                result["conditions"].append(record)
        order = [(m, route) for m in (2, 4, 8) for route in (0, 1)]
        for phase, count in (("warmup", args.warmup), ("measured", args.repetitions)):
            for repetition in range(count):
                for m, route in order if repetition % 2 == 0 else list(reversed(order)):
                    io, keep, outputs = groups[m]
                    os.environ[ROUTING] = str(route)
                    ort.copy_tensors(snapshots, state_values)
                    if phase == "warmup" and repetition == 0:
                        if digest(read(state_values)) != snapshot_hash:
                            raise AssertionError("Frozen target state was not restored exactly.")
                    read([keep["input_ids"]])
                    profile = args.output / f"c{length}-m{m}-on{route}-r{repetition}.jsonl"
                    if args.kind == "gpu" and phase == "measured":
                        os.environ[GPU_PROFILE] = str(profile)
                    start = time.perf_counter_ns()
                    try:
                        session.run_with_iobinding(io)
                        enqueue_ms = (time.perf_counter_ns() - start) / 1e6
                        logits = read([outputs["logits"]])[0]
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                    finally:
                        os.environ.pop(GPU_PROFILE, None)
                    aux = read([outputs["aux_hidden_states"]])[0]
                    if not np.isfinite(logits).all() or not np.isfinite(aux).all():
                        raise AssertionError("Nonfinite full-target output.")
                    prefix = args.output / f"c{length}-m{m}-on{route}"
                    if phase == "warmup" and repetition == 0:
                        np.save(str(prefix) + "-logits.npy", logits)
                        np.save(str(prefix) + "-aux.npy", aux)
                    else:
                        np.testing.assert_array_equal(logits, np.load(str(prefix) + "-logits.npy"))
                        np.testing.assert_array_equal(aux, np.load(str(prefix) + "-aux.npy"))
                    record = conditions[(m, route)]
                    if phase == "warmup":
                        record["warmup_host_ms"].append(elapsed)
                    else:
                        call = {
                            "repetition": repetition,
                            "host_ms": elapsed,
                            "enqueue_ms": enqueue_ms,
                            "logits_sha256": digest([logits]),
                        }
                        if args.kind == "gpu":
                            call["gpu"] = profile_summary(profile)
                            expected = 192 if route else 0
                            if call["gpu"]["programs"].get("DiagnosticSmallMSimt", 0) != expected:
                                raise AssertionError("The full target did not route exactly the 192 MLP projections.")
                        record["calls"].append(call)
                        print("controlled", args.kind, length, m, route, repetition, round(elapsed, 3), flush=True)
                    write_json(args.output / "result.json", result)
        for m in (2, 4, 8):
            prefix = args.output / f"c{length}-m{m}"
            a, b = (np.load(f"{prefix}-on{route}-logits.npy") for route in (0, 1))
            conditions[(m, 1)]["vs_existing_logits"] = {
                **errors(b, a),
                "different_elements": int(np.count_nonzero(a != b)),
                "argmax_different_rows": int(np.count_nonzero(np.argmax(a, -1) != np.argmax(b, -1))),
                "existing_argmax": np.argmax(a, -1).tolist(),
                "experimental_argmax": np.argmax(b, -1).tolist(),
            }
            a, b = (np.load(f"{prefix}-on{route}-aux.npy") for route in (0, 1))
            conditions[(m, 1)]["vs_existing_aux"] = errors(b, a)
        if digest(read(snapshots)) != snapshot_hash:
            raise AssertionError("The immutable snapshot changed.")
        del snapshots, arrays, groups
    if args.kind == "assignment":
        result["assignment"] = validate_assignment([Path(session.end_profiling())])
    result["mapped_binaries"] = mapped_binaries(args.plugin)
    result["gpu_sample"] = gpu_sample()
    result["status"] = "passed"
    write_json(args.output / "result.json", result)


def stats_delta(before, after, speculative):
    keys = (
        "rounds",
        "completed_rounds",
        "draft_tokens_proposed",
        "draft_tokens_evaluated",
        "draft_tokens_accepted",
        "target_forward_passes",
        "draft_forward_passes",
        "dflash2_failures",
        "dflash2_disables",
        "dflash2_admission_misses",
        "standard_fallback_steps",
    )
    counts = {key: after[key] - before[key] for key in keys}
    accepted, evaluated, proposed = (
        counts[key] for key in ("draft_tokens_accepted", "draft_tokens_evaluated", "draft_tokens_proposed")
    )
    failures = ("dflash2_failures", "dflash2_disables", "dflash2_admission_misses", "standard_fallback_steps")
    if min(counts.values()) < 0 or not 0 <= accepted <= evaluated <= proposed or any(counts[key] for key in failures):
        raise AssertionError(f"Invalid/fallback speculative execution: {counts}")
    if speculative != (counts["completed_rounds"] > 0 and counts["draft_forward_passes"] > 0):
        raise AssertionError(f"Wrong generation mode actually executed: {counts}")
    if not speculative and (counts["rounds"] or proposed):
        raise AssertionError("Baseline unexpectedly speculated.")
    counts["acceptance_rate"] = accepted / evaluated if evaluated else None
    counts["accepted_per_verification"] = accepted / counts["completed_rounds"] if counts["completed_rounds"] else None
    counts["acceptance_length_histogram"] = [
        new - old
        for old, new in zip(before["acceptance_length_histogram"], after["acceptance_length_histogram"], strict=True)
    ]
    return counts


def generate(og, engine, tokenizer, workload, speculative):
    tokens = np.asarray(workload["prompt_ids"], np.int32)
    requested = workload["output_tokens"]
    before = engine.get_speculative_stats()
    options = og.RequestOptions()
    options.set_max_session_tokens(len(tokens) + requested)
    request = engine.create_request(options=options)
    turn = og.TurnOptions(request)
    turn.set_do_sample(False)
    turn.set_min_generated_tokens(0)
    turn.set_max_generated_tokens(requested)
    turn.set_repetition_penalty(1.0)
    turn.set_no_repeat_ngram_size(0)
    turn.set_stop_strings([])
    turn.set_seed(0)
    buffer = engine.create_event_buffer(8)
    output, first, finish = [], None, None
    start = time.perf_counter()
    try:
        turn_id = request.begin_turn(tokens, turn)
        while engine.has_pending_requests():
            for event in engine.run(buffer):
                if event.flags & (
                    og.EngineEventFlags.FAILED | og.EngineEventFlags.RETRYABLE | og.EngineEventFlags.CAPACITY_BLOCKED
                ):
                    raise RuntimeError(f"Engine request failed: {event.flags}, {event.error_code}")
                if event.request is not request or event.turn_id != turn_id:
                    raise AssertionError("Event belongs to an unexpected request.")
                if event.flags & og.EngineEventFlags.TOKEN:
                    output.append(int(event.token))
                    if first is None:
                        first = time.perf_counter()
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    if finish is not None:
                        raise AssertionError("Duplicate completion event.")
                    finish = {
                        "reason": str(event.finish_reason),
                        "prompt_tokens": event.usage.prompt_tokens,
                        "generated_tokens": event.usage.generated_tokens,
                        "cached_prompt_tokens": event.usage.cached_prompt_tokens,
                    }
        elapsed = time.perf_counter() - start
    finally:
        request.close()
    if first is None or finish is None or len(output) != requested or finish["generated_tokens"] != requested:
        raise AssertionError(f"Incorrect generated length or early EOS: {len(output)}, {finish}")
    if finish["prompt_tokens"] != len(tokens) or finish["cached_prompt_tokens"] != 0:
        raise AssertionError("Prompt length changed or prefix caching was active.")
    if any(token < 0 or token >= 248320 for token in output):
        raise AssertionError("Out-of-vocabulary output.")
    return {
        "generation_seconds": elapsed,
        "ttft_seconds": first - start,
        "output_ids": output,
        "output_text": tokenizer.decode(np.asarray(output, np.int32)),
        "finish": finish,
        "speculative": stats_delta(before, engine.get_speculative_stats(), speculative),
    }


def gpu_sample():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def generation(args, ort, og):
    workloads = json.loads((args.runtime_root / "workloads.json").read_text())
    if [(len(w["prompt_ids"]), w["output_tokens"]) for w in workloads] != [(86, 128), (1315, 512)]:
        raise AssertionError("The established short/long workloads changed.")
    result = {**metadata(args, ort, og), "workloads": workloads, "warmups": [], "trials": [], "sessions": {}}
    sessions = {}
    for mode in args.modes:
        directory, width = MODES[mode]
        path = args.runtime_root / "models" / directory
        config_json = json.loads((path / "genai_config.json").read_text())
        dflash = config_json["model"]["dflash2"]
        exported_width = width or 7
        if dflash["num_draft_tokens"] != exported_width or dflash["block_size"] != exported_width + 1:
            raise AssertionError("A genuine width-specific DFlash2 export is required.")
        manifest = path / "export-manifest.json"
        if not manifest.is_file():
            raise AssertionError("Missing genuine DFlash2 export provenance.")
        overlay = {
            "model": {"mtp": {"enabled": False}},
            "search": {"min_length": 0, "num_beams": 1, "batch_size": 1},
            "engine": {"dynamic_batching": {"prefix_caching": False}},
            "speculative": {"max_draft_tokens": width or 7, "ngram_size": 0, "min_adaptive_k": 0, "cooldown": False},
        }
        if not width:
            overlay["model"]["dflash2"] = {"filename": ""}
        if args.kind == "routes":
            folder = args.output / mode
            folder.mkdir()
            overlay["model"]["decoder"] = {"session_options": {"enable_profiling": str(folder / "target")}}
            if width:
                overlay["model"]["dflash2"] = {"session_options": {"enable_profiling": str(folder / "draft")}}
        config = og.Config(str(path))
        config.clear_providers()
        config.append_provider("webgpu")
        for key, value in {"dawnBackendType": "Vulkan", "validationMode": "basic", "enableGraphCapture": "0"}.items():
            config.set_provider_option("webgpu", key, value)
        config.overlay(json.dumps(overlay))
        started = time.perf_counter()
        model = og.Model(config)
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        engine, tokenizer = og.Engine(model), og.Tokenizer(model)
        startup_seconds = time.perf_counter() - started
        for workload in workloads:
            rendered = tokenizer.apply_chat_template(
                messages=json.dumps([{"role": "user", "content": workload["prompt"]}]), add_generation_prompt=True
            )
            if [int(token) for token in tokenizer.encode(rendered)] != workload["prompt_ids"]:
                raise AssertionError("Prompt tokenizer/template mismatch.")
        sessions[mode] = config, model, engine, tokenizer
        result["sessions"][mode] = {
            "directory": str(path),
            "exported_draft_tokens": exported_width,
            "runtime_width": width,
            "query_rows": dflash["block_size"],
            "config_sha256": file_hash(path / "genai_config.json"),
            "target_onnx_sha256": file_hash(path / "model.onnx"),
            "drafter_onnx_sha256": file_hash(path / "dflash2.onnx"),
            "export_manifest_sha256": file_hash(manifest),
            "overlay": overlay,
            "model_load_seconds": load_seconds,
            "engine_startup_seconds": startup_seconds,
        }
        print("Loaded", mode, round(load_seconds, 3), "seconds; query rows", dflash["block_size"], flush=True)
        result["mapped_binaries"] = mapped_binaries(args.plugin)
        write_json(args.output / "result.json", result)
    config = model = engine = tokenizer = None
    order = [(mode, route) for mode in args.modes for route in (0, 1)]
    for workload in workloads:
        actual = {**workload, "output_tokens": 16} if args.kind == "routes" else workload
        for phase, count in (("warmups", 1), ("trials", args.repetitions)):
            for repetition in range(count):
                for mode, route in order if repetition % 2 == 0 else list(reversed(order)):
                    engine, tokenizer = sessions[mode][2], sessions[mode][3]
                    os.environ[ROUTING] = str(route)
                    profile = args.output / f"{workload['name']}-{mode}-on{route}.jsonl"
                    if args.kind == "routes" and phase == "trials":
                        os.environ[GPU_PROFILE] = str(profile)
                    try:
                        trial = generate(og, engine, tokenizer, actual, mode != "baseline")
                    finally:
                        os.environ.pop(GPU_PROFILE, None)
                    trial.update(
                        workload=workload["name"],
                        configuration=mode,
                        routing=route,
                        repetition=repetition,
                        gpu_sample=gpu_sample(),
                    )
                    if args.kind == "routes" and phase == "trials":
                        trial["gpu"] = profile_summary(profile)
                        routed = trial["gpu"]["programs"].get("DiagnosticSmallMSimt", 0)
                        if (routed > 0) != (route == 1 and mode != "baseline"):
                            raise AssertionError("Actual full-generation SIMT routing did not match the condition.")
                    result[phase].append(trial)
                    print(
                        phase,
                        workload["name"],
                        mode,
                        route,
                        repetition,
                        round(trial["generation_seconds"], 4),
                        trial["speculative"]["draft_tokens_accepted"],
                        trial["speculative"]["completed_rounds"],
                        flush=True,
                    )
                    write_json(args.output / "result.json", result)
    engine = tokenizer = None
    sessions.clear()
    gc.collect()
    if args.kind == "routes":
        result["assignment"] = {
            mode: validate_assignment(sorted((args.output / mode).glob("*.json"))) for mode in args.modes
        }
    result["status"] = "passed"
    write_json(args.output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME)
    parser.add_argument("--plugin", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("controlled")
    replay.add_argument("--state-dir", type=Path, default=RUNTIME.parent / "qwen38-verify-m2-m4-20261021")
    replay.add_argument("--kind", choices=("wall", "gpu", "assignment"), required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--warmup", type=int, default=8)
    replay.add_argument("--repetitions", type=int, default=10)
    replay.add_argument("--pci-bus", default="000e:00:00.0")
    engine = commands.add_parser("generation")
    engine.add_argument("--kind", choices=("timing", "routes"), required=True)
    engine.add_argument("--output", type=Path, required=True)
    engine.add_argument("--modes", choices=tuple(MODES), nargs="+", default=list(MODES))
    engine.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1 or getattr(args, "warmup", 1) < 1:
        parser.error("Warmup and repetitions must be positive.")
    args.plugin, args.runtime_root, args.output = (
        args.plugin.resolve(),
        args.runtime_root.resolve(),
        args.output.resolve(),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    ort, og = load_runtime(args)
    if args.command == "controlled":
        controlled(args, ort, og)
    else:
        generation(args, ort, og)


if __name__ == "__main__":
    main()
