# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Host-side reproducible CUDA Session performance experiment (no runtime edits)."""

import argparse
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import time
from pathlib import Path

import ml_dtypes
import numpy as np
from make_cuda_session_smoke import pack_rows
from onnx import TensorProto, checker, helper, numpy_helper, save_model
from onnx import __version__ as onnx_version

ROOT = Path(__file__).resolve().parents[5]
MODES = ("constant", "dynamic_metadata", "dynamic_all")


def command(arguments, **kwargs):
    return subprocess.check_output([str(value) for value in arguments], text=True, **kwargs).strip()


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def docker(args, arguments, **kwargs):
    return command(["docker", "exec", args.container, *arguments], **kwargs)


def compile_client(args):
    link = docker(args, ["ninja", "-C", args.build_dir, "-t", "commands", "onnxruntime_provider_test"]).splitlines()[-1]
    tokens = shlex.split(link)
    libraries = tokens[tokens.index("-o") + 2 :]
    if "&&" in libraries:
        libraries = libraries[: libraries.index("&&")]
    binary = args.work / "session_perf"
    source = Path(__file__).with_suffix(".cc")
    command(
        [
            "docker",
            "exec",
            "-w",
            args.build_dir,
            args.container,
            "g++",
            "-std=c++17",
            "-O2",
            "-Wl,--export-dynamic",
            f"-I{ROOT}/include/onnxruntime/core/session",
            "-I/usr/local/cuda/include",
            source,
            "-o",
            binary,
            *libraries,
            "-L/usr/local/cuda/lib64",
            "-lcudart",
            "-Wl,-rpath,/usr/local/cuda/lib64",
        ]
    )
    for library in ("libonnxruntime_providers_shared.so", "libonnxruntime_providers_cuda.so"):
        link = args.work / library
        if not link.is_symlink():
            link.symlink_to(args.build_dir / library)
    return binary


def fixtures(args, first_bits, second_bits, dtype_name):
    directory = args.work / f"{first_bits}{second_bits}_{dtype_name}"
    directory.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if dtype_name == "fp16" else ml_dtypes.bfloat16
    tensor_type = TensorProto.FLOAT16 if dtype_name == "fp16" else TensorProto.BFLOAT16
    hidden, inter, experts = args.hidden, args.intermediate, args.experts
    tensors = {
        "input": ((np.arange(hidden).reshape(1, 1, hidden) % 17 - 8) * 0.04).astype(dtype),
        "router": ((np.arange(experts).reshape(1, experts) % 23 - 11) * 0.125).astype(dtype),
    }
    for prefix, outputs, features, bits in (
        ("first", 2 * inter, hidden, first_bits),
        ("second", hidden, inter, second_bits),
    ):
        codes = ((np.arange(32)[:, None] * 11 + np.arange(features) * 7) % (1 << bits)).astype(np.uint8)
        tensors[prefix] = np.tile(pack_rows(codes, bits), (experts * outputs // 32, 1)).reshape(experts, outputs, -1)
        tensors[prefix + "_scale"] = np.full((experts, outputs, features // args.block), 0.015625, dtype=dtype)
        zeros = pack_rows(np.full((1, features // args.block), 1 << (bits - 1), dtype=np.uint8), bits)
        tensors[prefix + "_zeros"] = np.tile(zeros, (experts * outputs, 1)).reshape(experts, outputs, -1)
    initializers = {name: numpy_helper.from_array(values, name) for name, values in tensors.items()}
    metadata = {}
    for name, values in tensors.items():
        path = directory / f"{name}.bin"
        values.tofile(path)
        metadata[name] = {"bytes": path.stat().st_size, "sha256": digest(path), "shape": list(values.shape)}
    node = helper.make_node(
        "QMoE",
        [
            "input",
            "router",
            "first",
            "first_scale",
            "",
            "second",
            "second_scale",
            "",
            "",
            "",
            "",
            "first_zeros",
            "second_zeros",
            "",
        ],
        ["output"],
        domain="com.microsoft",
        k=args.top_k,
        activation_type="swiglu",
        swiglu_fusion=1,
        normalize_routing_weights=1,
        expert_weight_bits=first_bits,
        fc2_expert_weight_bits=second_bits,
        block_size=args.block,
        weights_prepacked=0,
    )
    for mode in MODES:
        dynamic = {"input", "router"}
        if mode == "dynamic_metadata":
            dynamic.update(name for name in tensors if "scale" in name or "zeros" in name)
        elif mode == "dynamic_all":
            dynamic.update(tensors)
        graph = helper.make_graph(
            [node],
            f"qmoe-perf-{mode}",
            [
                helper.make_tensor_value_info(name, initializers[name].data_type, values.shape)
                for name, values in tensors.items()
                if name in dynamic
            ],
            [helper.make_tensor_value_info("output", tensor_type, [1, 1, hidden])],
            [tensor for name, tensor in initializers.items() if name not in dynamic],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 10
        checker.check_model(model)
        save_model(model, directory / f"{mode}.onnx")
        lines = [
            f"{name} {directory / (name + '.bin')} {initializers[name].data_type} {values.ndim} "
            + " ".join(map(str, values.shape))
            for name, values in tensors.items()
            if name in dynamic
        ]
        (directory / f"{mode}.feeds").write_text("\n".join(lines) + "\n")
    return directory, metadata


def summarize(records):
    summary = {}
    for mode in MODES:
        runs = [record for record in records if record["mode"] == mode]
        steady = [sample for record in runs for sample in record["steady_ms"]]
        summary[mode] = {
            "init_median_ms": statistics.median(record["init_ms"] for record in runs),
            "first_run_median_ms": statistics.median(record["first_run_ms"] for record in runs),
            "steady_median_ms": statistics.median(steady),
            "steady_min_ms": min(steady),
            "steady_max_ms": max(steady),
            "per_process_steady_medians_ms": [statistics.median(record["steady_ms"]) for record in runs],
            "resident_increment_median_bytes": statistics.median(
                record["device_used_bytes"]["after_steady"] - record["device_used_bytes"]["before_feeds"]
                for record in runs
            ),
        }
    constant = summary["constant"]
    for mode in MODES[1:]:
        summary[mode]["steady_speedup_constant_over_mode"] = (
            summary[mode]["steady_median_ms"] / constant["steady_median_ms"]
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="jiafa-dev")
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/int3-pr2-pr3-clean/Release"))
    parser.add_argument(
        "--work", type=Path, required=True, help="Scratch path mounted at the same path in the container"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--quick", action="store_true", help="Only 3/3 FP16; dimensions still explicit")
    args = parser.parse_args()
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert args.hidden % args.block == args.intermediate % args.block == 0
    assert (2 * args.intermediate * args.experts) % 32 == (args.hidden * args.experts) % 32 == 0
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    binary = compile_client(args)
    loader = ":".join(
        [
            str(args.build_dir),
            "/usr/local/cuda/lib64",
            "/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib",
            "/usr/local/lib/python3.12/dist-packages/torch/lib",
        ]
    )
    report = {
        "schema_version": 1,
        "started_utc": started,
        "command": shlex.join(["python3", *os.sys.argv]),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "git_head": command(["git", "-C", ROOT, "rev-parse", "HEAD"]),
        "git_status": command(["git", "-C", ROOT, "status", "--short"]),
        "onnx_source_head": command(
            ["git", "-C", ROOT.parent / "onnxruntime/cmake/external/onnx", "rev-parse", "HEAD"]
        ),
        "onnx_pinned_gitlink": command(["git", "-C", ROOT, "ls-tree", "HEAD", "cmake/external/onnx"]),
        "python_dependencies": {"numpy": np.__version__, "onnx": onnx_version, "ml_dtypes": ml_dtypes.__version__},
        "gpu_inventory": command(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.used", "--format=csv"]
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in [
                Path(__file__),
                Path(__file__).with_suffix(".cc"),
                ROOT / "onnxruntime/contrib_ops/cuda/moe/qmoe_int3.cc",
                ROOT / "onnxruntime/contrib_ops/cuda/moe/qmoe_int3.cu",
                ROOT / "onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc",
            ]
        },
        "binary_sha256": docker(
            args,
            [
                "sha256sum",
                binary,
                args.build_dir / "libonnxruntime_session.a",
                args.build_dir / "libonnxruntime_providers_cuda.so",
                args.build_dir / "CMakeCache.txt",
            ],
        ),
        "cases": [],
    }
    for first, second in ((3, 3),) if args.quick else ((3, 3), (3, 4), (4, 3)):
        for dtype in ("fp16",) if args.quick else ("fp16", "bf16"):
            directory, tensors = fixtures(args, first, second, dtype)
            records = []
            reference = None
            max_error = 0.0
            for repeat in range(args.repeats):
                order = MODES if repeat % 2 == 0 else tuple(reversed(MODES))
                for position, mode in enumerate(order):
                    result = directory / f"{mode}.{repeat}.json"
                    docker(
                        args,
                        [
                            "env",
                            "CUDA_VISIBLE_DEVICES=0",
                            f"LD_LIBRARY_PATH={loader}",
                            binary,
                            directory / f"{mode}.onnx",
                            directory / f"{mode}.feeds",
                            result,
                            args.warmup,
                            args.samples,
                        ],
                    )
                    record = json.loads(result.read_text())
                    record.update(mode=mode, repeat=repeat, position=position)
                    records.append(record)
                    output = np.fromfile(str(result) + ".output.f32", dtype=np.float32)
                    if reference is None:
                        reference = output
                    assert reference.shape == output.shape and np.isfinite(output).all()
                    max_error = max(max_error, float(np.max(np.abs(reference - output))))
                    np.testing.assert_allclose(output, reference, rtol=0, atol=0)
                    print(
                        f"{first}/{second} {dtype} {repeat} {mode}: init={record['init_ms']:.3f} first={record['first_run_ms']:.3f} steady={statistics.median(record['steady_ms']):.3f} ms",
                        flush=True,
                    )
            report["cases"].append(
                {
                    "bits": [first, second],
                    "dtype": dtype,
                    "tensor_provenance": tensors,
                    "output_max_abs_difference": max_error,
                    "output_nonzero_count": int(np.count_nonzero(reference)),
                    "records": records,
                    "summary": summarize(records),
                }
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
