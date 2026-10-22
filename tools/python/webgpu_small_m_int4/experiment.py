# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Correctness and resident-weight timestamp experiments for diagnostic SIMT INT4."""

import argparse
import copy
import datetime
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from reference import assert_fp32_accuracy, errors, pack_nibbles, pack_zero_points, projection

SHAPES = {(5120, 1024), (5120, 10240), (5120, 17408), (17408, 5120)}
MS = (1, 2, 3, 4, 5, 7, 8)


@dataclass
class Case:
    name: str
    a: np.ndarray
    b: np.ndarray
    scales: np.ndarray
    block_size: int = 32
    zero_points: np.ndarray | None = None
    bias: np.ndarray | None = None
    role: str = "synthetic"
    initializer_protos: list | None = None
    source_node: str | None = None
    activation_source: str = "deterministic synthetic FP16"

    @property
    def k(self):
        return self.a.shape[-1]

    @property
    def n(self):
        return self.b.shape[0]

    @property
    def m(self):
        return self.a.size // self.k

    @property
    def output_shape(self):
        return (*self.a.shape[:-1], self.n)

    def metadata(self):
        return {
            "name": self.name,
            "M": self.m,
            "K": self.k,
            "N": self.n,
            "input_shape": self.a.shape,
            "block_size": self.block_size,
            "zero_points": self.zero_points is not None,
            "bias": self.bias is not None,
            "role": self.role,
            "source_node": self.source_node,
            "activation_source": self.activation_source,
        }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def synthetic_case(name, m, k, n, block_size=32, zero=False, bias=False, rank=None, pattern="random"):
    rng = np.random.default_rng(3077 + m * 1000 + k + n)
    per_column = k if block_size == k * n else block_size
    blocks = (k + per_column - 1) // per_column
    q = rng.integers(0, 16, (n, blocks, per_column), np.uint8)
    scales = rng.uniform(0.001, 0.017, (n, blocks)).astype(np.float16)
    if block_size == k * n:
        scales = scales.ravel()[:1]
    zeros = rng.integers(0, 16, (n, blocks), np.uint8) if zero else None
    bias_values = rng.uniform(-0.1, 0.1, n).astype(np.float16) if bias else None
    a = rng.normal(0, 0.25, (m, k)).astype(np.float16)
    if pattern == "nibbles":
        q[:] = np.arange(q.size, dtype=np.uint64).reshape(q.shape) % 16
        scales.fill(0.125)
        a[:] = np.linspace(-0.5, 0.5, k, dtype=np.float16)
    elif pattern == "cancellation":
        q[:] = np.where(np.arange(blocks)[None, :, None] % 8 < 4, 15, 1)
        scales.fill(0.125)
        a.fill(256)
    elif pattern == "magnitudes":
        a = (rng.choice([-1, 1], (m, k)) * np.exp2(rng.integers(-8, 7, (m, k)))).astype(np.float16)
        scales = (rng.choice([-1, 1], (n, blocks)) * np.exp2(rng.integers(-14, 0, (n, blocks)))).astype(np.float16)
    elif pattern == "small":
        a.fill(2**-8)
        scales.fill(2**-14)
    elif pattern == "zeros":
        a.fill(0)
        scales.fill(0)
    if rank is not None:
        a = a.reshape(*rank, k)
    return Case(
        name,
        a,
        pack_nibbles(q),
        scales,
        block_size,
        pack_zero_points(zeros) if zeros is not None else None,
        bias_values,
    )


def synthetic_cases(smoke=False):
    if smoke:
        return [
            synthetic_case("multirow", 4, 96, 17, zero=True, bias=True),
            synthetic_case("partial_m_k", 3, 513, 31, zero=True),
        ]
    result = [
        synthetic_case(f"m{m}_k{k}_n{n}", m, k, n, zero=(i + m) % 3 == 0, bias=(i + m) % 4 == 0)
        for m in MS
        for i, (k, n) in enumerate(((32, 1), (96, 17), (100, 65), (513, 31), (1056, 33), (5120, 64), (17408, 65)))
    ]
    result += [synthetic_case(f"k_tail_{k}", 3, k, 15, zero=True) for k in (1, 7, 31, 33, 63, 95, 511, 512, 544)]
    result += [synthetic_case(f"n_tail_{n}", 4, 64, n, bias=True) for n in (7, 8, 15, 16, 31, 32)]
    result += [
        synthetic_case("block64", 5, 100, 17, block_size=64, zero=True, bias=True),
        synthetic_case("block128", 7, 513, 33, block_size=128, zero=True),
        synthetic_case("rank3", 4, 100, 33, zero=True, rank=(2, 2)),
        synthetic_case("rank4", 8, 96, 17, bias=True, rank=(2, 1, 4)),
        synthetic_case("vector", 1, 63, 7, zero=True, rank=()),
        synthetic_case("all_nibbles", 4, 96, 17, zero=True, pattern="nibbles"),
        synthetic_case("cancel5120", 4, 5120, 17, pattern="cancellation"),
        synthetic_case("cancel17408", 4, 17408, 17, pattern="cancellation"),
        synthetic_case("large_small", 4, 1056, 33, pattern="magnitudes"),
        synthetic_case("small_products", 8, 5120, 17, pattern="small"),
        synthetic_case("zero_values", 5, 513, 17, zero=True, bias=True, pattern="zeros"),
        synthetic_case("single_scale", 4, 128, 16, block_size=2048, zero=True, bias=True),
        synthetic_case("normalized_dispatch", 3, 32, 1048577),
    ]
    result[0].scales = result[0].scales.ravel()
    return result


def write_model(cases, directory):
    directory.mkdir(parents=True, exist_ok=True)
    nodes, inputs, outputs, initializers = [], [], [], []
    vi, fp16 = onnx.helper.make_tensor_value_info, onnx.TensorProto.FLOAT16
    external_protos = {}
    if any(case.initializer_protos is not None for case in cases):
        # Keep a private regular file: ONNX rejects external data with multiple hard links.
        with (directory / "weights.bin").open("wb") as stream:
            for i, case in enumerate(cases):
                if case.initializer_protos is None:
                    continue
                arrays = [array for array in (case.b, case.scales, case.zero_points, case.bias) if array is not None]
                external_protos[i] = []
                for original, array in zip(case.initializer_protos, arrays, strict=True):
                    tensor = copy.deepcopy(original)
                    offset = stream.tell()
                    array.tofile(stream)
                    tensor.raw_data = b""
                    onnx.external_data_helper.set_external_data(
                        tensor, "weights.bin", offset=offset, length=array.nbytes
                    )
                    tensor.ClearField("raw_data")
                    external_protos[i].append(tensor)
    for i, case in enumerate(cases):
        names = [f"A{i}", f"B{i}", f"S{i}"]
        tensors = [case.b, case.scales]
        tensor_names = names[1:]
        if case.zero_points is not None:
            names.append(f"Z{i}")
            tensors.append(case.zero_points)
            tensor_names.append(f"Z{i}")
        if case.bias is not None:
            if case.zero_points is None:
                names.append("")
            names.extend(["", f"bias{i}"])
            tensors.append(case.bias)
            tensor_names.append(f"bias{i}")
        if case.initializer_protos is None:
            initializers.extend(
                onnx.numpy_helper.from_array(t, name) for t, name in zip(tensors, tensor_names, strict=True)
            )
        else:
            for original, name in zip(external_protos[i], tensor_names, strict=True):
                tensor = copy.deepcopy(original)
                tensor.name = name
                initializers.append(tensor)
        nodes.append(
            onnx.helper.make_node(
                "MatMulNBits",
                names,
                [f"Y{i}"],
                name=case.name,
                domain="com.microsoft",
                K=case.k,
                N=case.n,
                bits=4,
                block_size=case.block_size,
                accuracy_level=4,
            )
        )
        inputs.append(vi(f"A{i}", fp16, case.a.shape))
        outputs.append(vi(f"Y{i}", fp16, case.output_shape))
        case.a.tofile(directory / f"input-A{i}.bin")
    model = onnx.helper.make_model(
        onnx.helper.make_graph(nodes, "small_m_simt_diagnostic", inputs, outputs, initializers),
        opset_imports=[onnx.helper.make_opsetid("", 21), onnx.helper.make_opsetid("com.microsoft", 1)],
        ir_version=9,
    )
    model_path = directory / "model.onnx"
    onnx.save(model, model_path)
    onnx.checker.check_model(str(model_path), full_check=False)
    write_json(directory / "cases.json", [c.metadata() for c in cases])
    return model_path


def invoke(
    args, model, prefix, mode, warmup, repeat, *, fp32=False, validation="full", dump=False, require_gpu_timing=True
):
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "ORT_WEBGPU_SMALL_M_BN": str(args.bn),
        "ORT_WEBGPU_SMALL_M_BK": str(args.bk),
        "ORT_WEBGPU_SMALL_M_A_BLOCK_PADDING": str(args.a_block_padding),
    }
    if args.icd is not None:
        env["VK_ICD_FILENAMES"] = str(args.icd)
        env.pop("VK_DRIVER_FILES", None)
    if dump:
        env["ORT_WEBGPU_EP_SHADER_DUMP_FILE"] = str(prefix) + ".wgsl"
    if getattr(args, "stats_library", None) is not None:
        env["LD_PRELOAD"] = str(args.stats_library.resolve())
        env["PIPELINE_STATS_PREFIX"] = str(prefix) + "-pipeline-stats"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.executable),
        str(model),
        str(model.parent / "input"),
        str(prefix),
        str(warmup),
        str(repeat),
        validation,
        str(int(fp32)),
        mode,
    ]
    with Path(str(prefix) + "-clocks.csv").open("w") as clock_file:
        monitor = (
            subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                    "--loop-ms=50",
                ],
                stdout=clock_file,
                stderr=subprocess.STDOUT,
            )
            if args.monitor_nvidia
            else None
        )
        try:
            process = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        finally:
            if monitor is not None:
                if monitor.poll() is not None:
                    raise RuntimeError(f"NVIDIA clock monitor exited prematurely: {monitor.returncode}")
                monitor.terminate()
                monitor.wait()
    text = process.stdout + process.stderr
    Path(str(prefix) + ".log").write_text(text)
    if process.returncode:
        raise RuntimeError(f"Diagnostic failed ({process.returncode}): {text}")
    profile = Path(next(line.removeprefix("PROFILE=") for line in text.splitlines() if line.startswith("PROFILE=")))
    events = json.loads(profile.read_text())
    nodes = [e for e in events if e.get("cat") == "Node" and e.get("ph") == "X"]
    if not nodes or any(e["args"].get("provider") != "WebGpuExecutionProvider" for e in nodes):
        raise AssertionError("The diagnostic did not exclusively execute nodes on WebGPU.")
    kernels = [e for e in events if e.get("cat") == "Api" and e.get("ph") == "X"]
    if require_gpu_timing and not kernels:
        raise AssertionError("No native GPU timestamp events were captured.")
    programs = (
        Counter(e["name"].split("&")[-1] for e in kernels)
        if kernels
        else Counter(re.findall(r'Starting program "([^[:"]+)', text))
    )
    if not programs:
        raise AssertionError("No actual program dispatches were observed.")
    if mode in ("simt", "regular"):
        expected = "MatMulNBitsSmallMSimt" if mode == "simt" else "MatMulNBits"
        if set(programs) != {expected}:
            raise AssertionError(f"Unexpected GPU dispatches: {programs}")
    stamps = {key: int(value) for key, value in re.findall(r"(MEASURE_START|MEASURE_END)=(\d+)", text)}
    clocks = []
    for line in Path(str(prefix) + "-clocks.csv").read_text().splitlines():
        fields = line.split(",")
        if len(fields) != 5:
            raise ValueError(f"Invalid clock sample: {line}")
        timestamp = datetime.datetime.strptime(fields[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
        timestamp = timestamp.timestamp() * 1e6
        if stamps["MEASURE_START"] <= timestamp <= stamps["MEASURE_END"]:
            clocks.append([int(value) for value in fields[1:]])
    result = {
        "command": command,
        "mode": mode,
        "fp32_option": fp32,
        "warmup": warmup,
        "repeat": repeat,
        "programs": dict(programs),
        "gpu_timestamps_available": bool(kernels),
        "profile": str(profile),
        "clocks_sm_mem_util_temp": clocks,
        "clock_monitor": "nvidia-smi" if args.monitor_nvidia else "unavailable; not monitored",
        "measurement_timestamps": stamps,
        "BN": args.bn,
        "BK": args.bk,
        "a_block_padding": args.a_block_padding,
    }
    write_json(Path(str(prefix) + ".json"), result)
    return result, kernels, np.fromfile(str(prefix) + "-outputs.bin", np.float16)


def split_outputs(cases, flat):
    result, offset = [], 0
    for case in cases:
        count = int(np.prod(case.output_shape))
        result.append(flat[offset : offset + count].reshape(case.output_shape))
        offset += count
    if offset != flat.size:
        raise ValueError("Diagnostic output size does not match the model.")
    return result


def correctness(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cases = synthetic_cases(args.smoke)
    model = write_model(cases, args.output / "model")
    values, runs = {}, {}
    for mode in ("simt", "regular", "regular-f32"):
        native_mode = "regular" if mode == "regular-f32" else mode
        run, _, flat = invoke(
            args,
            model,
            args.output / mode,
            native_mode,
            1,
            1,
            fp32=mode == "regular-f32",
            dump=mode == "simt",
            require_gpu_timing=False,
        )
        values[mode] = split_outputs(cases, flat)
        runs[mode] = run
    records = []
    for i, case in enumerate(cases):
        full, rounded, absolute = projection(
            case.a,
            case.b,
            case.scales,
            case.k,
            case.block_size,
            case.zero_points,
            case.bias,
        )
        assert_fp32_accuracy(values["simt"][i], rounded, absolute, case.k)
        record = {
            **case.metadata(),
            "comparisons": {
                mode: {
                    "vs_fp64": errors(values[mode][i], full),
                    "vs_fp64_half_weights": errors(values[mode][i], rounded),
                }
                for mode in values
            },
            "simt_vs_regular": errors(values["simt"][i], values["regular"][i]),
            "simt_vs_regular_f32": errors(values["simt"][i], values["regular-f32"][i]),
        }
        records.append(record)
        print(case.name, "max_abs=", record["comparisons"]["simt"]["vs_fp64_half_weights"]["max_abs"], flush=True)
    write_json(
        args.output / "result.json",
        {
            "status": "passed",
            "cases": records,
            "runs": runs,
            "reference": "FP64 byte-level nibble extraction, scale multiplication and dot products; separate FP16-weight reference.",
        },
    )


def external_array(tensor, source_dir):
    if not tensor.external_data:
        return onnx.numpy_helper.to_array(tensor)
    info = onnx.external_data_helper.ExternalDataInfo(tensor)
    return np.memmap(
        source_dir / info.location,
        mode="r",
        dtype=onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type),
        offset=info.offset or 0,
        shape=tuple(tensor.dims),
    )


def role_for(node, k, n):
    if (k, n) == (5120, 17408):
        return "mlp_gate" if "/gate_proj/" in node.name else "mlp_up"
    if (k, n) == (17408, 5120):
        return "mlp_down"
    return "attention_kv" if n == 1024 else "linear_qkv"


def real_cases(args, m):
    model = onnx.load(args.weights_model, load_external_data=False)
    tensors = {t.name: t for t in model.graph.initializer}
    capture_files = {}
    for rows in (2, 4):
        folder = args.captures / f"capture-m{rows}"
        capture_files[rows] = sorted(folder.glob("*-A.bin"), key=lambda p: int(p.name.split("-")[0]))
    groups = Counter()
    result = []
    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        k, n = attrs["K"], attrs["N"]
        if (k, n) not in SHAPES:
            continue
        if attrs["bits"] != 4 or attrs["block_size"] != 32 or len(node.input) != 3:
            raise ValueError(f"Unexpected real quantization layout: {node.name}")
        role = role_for(node, k, n)
        if args.matrices and groups[role] >= args.matrices:
            continue
        groups[role] += 1
        protos = [copy.deepcopy(tensors[name]) for name in node.input[1:]]
        arrays = [external_array(t, args.weights_model.parent) for t in protos]
        activation_source = "deterministic synthetic FP16"
        a = np.random.default_rng(k + n + m + len(result)).normal(0, 0.25, (m, k)).astype(np.float16)
        if role.startswith("mlp_"):
            layer = int(re.search(r"layers\.(\d+)", node.name)[1])
            captured_rows = min(m, 4)
            available = capture_files[captured_rows]
            if len(available) != 64:
                raise ValueError(f"Expected all 64 captured M={captured_rows} MLP activations.")
            path = available[layer]
            if role == "mlp_down":
                path = path.with_name(path.name.replace("-A.bin", "-Y.bin"))
            captured = np.fromfile(path, np.float16).reshape(captured_rows, k)
            a = captured if m <= 4 else np.tile(captured, (2, 1))
            activation_source = str(path) + ("; repeated twice for M=8, not an M=8 target capture" if m == 8 else "")
        result.append(
            Case(
                f"projection_{len(result)}",
                a,
                arrays[0],
                arrays[1],
                role=role,
                initializer_protos=protos,
                source_node=node.name,
                activation_source=activation_source,
            )
        )
    if set(groups) != {"mlp_gate", "mlp_up", "mlp_down", "attention_kv", "linear_qkv"}:
        raise ValueError(f"Missing requested real projection groups: {groups}")
    return result


def summarize_samples(cases, events, warmup, repeat):
    by_node = defaultdict(list)
    for event in events:
        by_node[event["name"].split("&")[0]].append(event)
    groups, counts, paths = {}, Counter(), {}
    for case in cases:
        entries = by_node.pop(case.name)
        calls = warmup + repeat
        if len(entries) % calls:
            raise AssertionError(f"Dispatches do not form complete rotations: {case.name}")
        stages = len(entries) // calls
        paths[case.name] = [e["name"].split("&")[-1] for e in entries[:stages]]
        samples = np.asarray([e["dur"] for e in entries], np.float64).reshape(calls, stages).sum(axis=1)[warmup:]
        if not np.all(samples > 0):
            raise AssertionError(f"Nonpositive GPU timestamp duration: {case.name}")
        groups.setdefault(case.role, np.zeros(repeat, np.float64))
        groups[case.role] += samples
        counts[case.role] += 1
    if by_node:
        raise AssertionError(f"Unaccounted GPU programs: {list(by_node)}")
    return {
        role: {
            "matrices": counts[role],
            "rotation_samples_us": samples.tolist(),
            "per_projection_median_us": float(np.median(samples / counts[role])),
            "per_projection_p10_us": float(np.percentile(samples / counts[role], 10)),
            "per_projection_p90_us": float(np.percentile(samples / counts[role], 90)),
            "rotation_median_ms": float(np.median(samples) / 1000),
        }
        for role, samples in groups.items()
    }, paths


def benchmark(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "weights_model": str(args.weights_model),
        "BN": args.bn,
        "BK": args.bk,
        "a_block_padding": args.a_block_padding,
        "conditions": [],
        "hardware": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
            text=True,
        )
        if args.monitor_nvidia
        else "See native runner logs for the selected adapter; clocks not monitored.",
    }
    for mi, m in enumerate(args.m):
        directory = args.output / f"m{m}" / "model"
        cases = real_cases(args, m)
        model = write_model(cases, directory)
        condition = {"M": m, "cases": [c.metadata() for c in cases], "runs": {}, "numerics": []}
        outputs = {}
        order = args.modes[mi % len(args.modes) :] + args.modes[: mi % len(args.modes)]
        for mode in order:
            native_mode = "regular" if mode == "regular-f32" else mode
            prefix = args.output / f"m{m}" / mode
            run, events, flat = invoke(
                args,
                model,
                prefix,
                native_mode,
                args.warmup,
                args.repeat,
                fp32=mode == "regular-f32",
                validation=args.validation,
                dump=mode == "simt",
            )
            run["groups"], run["paths"] = summarize_samples(cases, events, args.warmup, args.repeat)
            condition["runs"][mode] = run
            outputs[mode] = split_outputs(cases, flat)
            for case, values in zip(cases, outputs[mode], strict=True):
                if not np.isfinite(values).all():
                    raise AssertionError(f"{mode} produced nonfinite real outputs for {case.source_node}.")
            print(
                "M",
                m,
                mode,
                {role: round(data["per_projection_median_us"], 3) for role, data in run["groups"].items()},
                flush=True,
            )
        selected = set()
        by_role = defaultdict(list)
        for i, case in enumerate(cases):
            by_role[case.role].append(i)
        for indices in by_role.values():
            selected.update((indices[0], indices[len(indices) // 2], indices[-1]))
        for i, case in enumerate(cases):
            record = case.metadata()
            if "simt" in outputs and "regular" in outputs:
                record["simt_vs_regular"] = errors(outputs["simt"][i], outputs["regular"][i])
            if args.reference and i in selected:
                full, rounded, absolute = projection(case.a, case.b, case.scales, case.k, case.block_size)
                if "simt" in outputs:
                    assert_fp32_accuracy(outputs["simt"][i], rounded, absolute, case.k)
                record["reference"] = {
                    mode: {"vs_fp64": errors(values[i], full), "vs_fp64_half_weights": errors(values[i], rounded)}
                    for mode, values in outputs.items()
                }
            condition["numerics"].append(record)
        report["conditions"].append(condition)
        write_json(args.output / "result.json", report)
    report["status"] = "passed"
    write_json(args.output / "result.json", report)


def resources(args):
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for m in (2, 4, 8):
        case = synthetic_case("projection_0", m, 5120, 1024)
        model = write_model([case], args.output / f"m{m}" / "model")
        for mode in ("simt", "regular", "regular-f32", "auto"):
            native_mode = "regular" if mode == "regular-f32" else mode
            prefix = args.output / f"m{m}" / mode
            run, _, _ = invoke(args, model, prefix, native_mode, 2, 1, fp32=mode == "regular-f32", dump=True)
            records.append({"M": m, "variant": mode, **run, "statistics": str(prefix) + "-pipeline-stats.txt"})
    write_json(
        args.output / "result.json",
        {
            "runs": records,
            "note": "Instrumented compiler-statistics runs; do not use these timings as clean benchmarks.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True, help="Normal CMake MatMulNBits benchmark target.")
    parser.add_argument("--icd", type=Path, help="Optional Vulkan ICD override; unused on D3D12.")
    parser.add_argument("--monitor-nvidia", action="store_true", help="Collect NVIDIA clocks using nvidia-smi.")
    parser.add_argument("--bn", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--bk", type=int, choices=(256, 512, 1024), default=512)
    parser.add_argument("--a-block-padding", type=int, choices=(0, 1), default=0)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("correctness")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--smoke", action="store_true")
    bench = commands.add_parser("benchmark")
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--weights-model", type=Path, required=True)
    bench.add_argument("--captures", type=Path, required=True)
    bench.add_argument("--m", type=int, nargs="+", choices=(2, 4, 8), default=[2, 4, 8])
    bench.add_argument(
        "--modes", nargs="+", choices=("simt", "regular", "regular-f32", "auto"), default=["regular", "simt", "auto"]
    )
    bench.add_argument(
        "--matrices", type=int, default=0, help="Matrices per group; zero uses all available real weights."
    )
    bench.add_argument("--warmup", type=int, default=32)
    bench.add_argument("--repeat", type=int, default=15)
    bench.add_argument("--reference", action="store_true")
    bench.add_argument("--validation", choices=("full", "disabled"), default="full")
    stats = commands.add_parser("resources")
    stats.add_argument("--output", type=Path, required=True)
    stats.add_argument("--stats-library", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "correctness":
        correctness(args)
    elif args.command == "resources":
        resources(args)
    else:
        if args.matrices < 0 or args.warmup < 1 or args.repeat < 1:
            parser.error("Expected nonnegative matrices, positive warmup, and positive repeat.")
        benchmark(args)


if __name__ == "__main__":
    main()
