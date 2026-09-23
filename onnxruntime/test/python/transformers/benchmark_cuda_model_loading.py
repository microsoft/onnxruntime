# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Compare pageable, pinned and direct-storage external weights on one CUDA device.

Generate a deterministic, 4-KiB-aligned fixture and benchmark it (no PyTorch needed):
  python benchmark_cuda_model_loading.py --generate-model benchmark_data --repetitions 5

Reuse a model with independently computed reference outputs (NPZ keys are tensor names):
  python benchmark_cuda_model_loading.py --model model.onnx --inputs inputs.npz \
      --expected-outputs expected.npz --output results.json

Every warmup and measured sample runs in a fresh process. Only InferenceSession
construction is timed; imports, input loading, and correctness checks are excluded.
Dependencies are imported lazily so CLI help and reporting helpers need no ORT build.
Initialization completes weight transfers; a blocking inference and NumPy comparison
then verify the result. Loader INFO logs, not requested provider options, determine
the observed path. Direct storage means GDS on Linux or DirectStorage on Windows.
Uncompressed DirectStorage uses host/upload staging with a shared D3D12/CUDA
destination, not zero-host-copy NVMe-to-VRAM DMA. GDS requests native I/O with cuFile
compatibility mode disabled; the physical path remains platform-dependent.
ORT logs prove API use, not the physical storage-to-GPU DMA route.

Caches remain OS-managed (warmups and fixture creation can warm them). The optional
legacy --evict-file-cache is only a per-file, unprivileged POSIX hint, never proof
of cold storage. No privileged or global cache flushing is performed.
"""

import argparse
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

LOADER_LOG = re.compile(r"CUDA external data loader: path=(pageable|pinned|gds|directstorage) bytes=(\d+)\b")
UTF16LE_ASCII_LOG = re.compile(r"(?:[\t\r\n\x20-\x7e]\x00){2,}")
RESULT_PREFIX = "CUDA_LOADING_RESULT="
PATHS = ("pageable", "pinned", "direct")
PATH_DESCRIPTIONS = {
    "pageable": "CPU pageable buffer followed by H2D copy",
    "pinned": "CPU pinned staging buffers followed by H2D copy",
    "gds": (
        "NVIDIA cuFile API; native GDS requested with compatibility mode disabled, "
        "physical storage-to-GPU path remains platform-dependent"
    ),
    "directstorage": (
        "Microsoft DirectStorage API; uncompressed host/upload staging with a shared D3D12/CUDA destination, "
        "not zero-host-copy NVMe-to-VRAM DMA"
    ),
}


def reading_thread_count(value):
    count = int(value)
    if not 0 <= count <= 64:
        raise argparse.ArgumentTypeError("must be between 0 and 64")
    return count


def positive_int(value):
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return count


def nonnegative_int(value):
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return count


def evict_file_cache(path):
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("file-cache eviction requires os.posix_fadvise")
    with open(path, "rb") as file:
        os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def direct_path(platform_name):
    if platform_name.startswith("linux"):
        return "gds"
    if platform_name == "win32":
        return "directstorage"
    return None


def provider_options(path, device_id, reading_threads, platform_name):
    options = {
        "device_id": device_id,
        "external_data_loader_reading_threads": 0 if path == "pageable" else reading_threads,
        "external_data_loader_use_gds": 0,
        "external_data_loader_use_directstorage": 0,
        "use_tf32": 0,
    }
    if path == "direct":
        api_path = direct_path(platform_name)
        if api_path is None:
            raise ValueError(f"Direct storage is unsupported on {platform_name}")
        options[f"external_data_loader_use_{api_path}"] = 1
    return options


def parse_loader_logs(stderr):
    """Parse mixed Python text and Windows native UTF-16LE ASCII log segments.

    Only paired ASCII/NUL runs are normalized, not arbitrary NUL characters.
    The caller retains the original stderr as evidence in the sample.
    """
    stderr = UTF16LE_ASCII_LOG.sub(lambda match: match[0][::2], stderr)
    observed = {}
    for path, size in LOADER_LOG.findall(stderr):
        observed[path] = observed.get(path, 0) + int(size)
    warnings = [
        line
        for line in stderr.splitlines()
        if "CUDA external data loader" in line
        and ("fallback" in line.lower() or "falling back" in line.lower() or "[W:" in line)
    ]
    return observed, warnings


def classify_path(expected_path, observed, expected_bytes):
    """Require complete byte accounting before claiming the requested path ran."""
    observed = {path: size for path, size in observed.items() if size > 0}
    if not observed:
        return "unverified"
    if sum(observed.values()) != expected_bytes:
        return "incomplete"
    if len(observed) > 1:
        return "mixed"
    if expected_path not in observed:
        return "fallback"
    return "verified"


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_samples(samples, expected_bytes):
    """Keep verified, mixed, and fallback measurements in separate distributions."""
    summaries = []
    groups = {}
    for sample in samples:
        key = (sample["requested_path"], sample["status"], tuple(sorted(sample["observed_bytes"].items())))
        groups.setdefault(key, []).append(sample)
    for (requested, status, observed), group in groups.items():
        seconds = [sample["seconds"] for sample in group]
        median = statistics.median(seconds)
        summaries.append(
            {
                "requested_path": requested,
                "status": status,
                "observed_bytes": dict(observed),
                "observed_path_descriptions": {path: PATH_DESCRIPTIONS[path] for path, _ in observed},
                "throughput_definition": (
                    "External weight bytes divided by end-to-end InferenceSession initialization time; "
                    "not storage bandwidth"
                ),
                "count": len(group),
                "seconds": {
                    "min": min(seconds),
                    "median": median,
                    "mean": statistics.mean(seconds),
                    "stdev": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
                    "p90": percentile(seconds, 0.9),
                    "p95": percentile(seconds, 0.95),
                    "max": max(seconds),
                },
                # End-to-end initialization throughput, not a device or disk bandwidth measurement.
                "effective_gib_per_second": {
                    "median": statistics.median(expected_bytes / (2**30) / value for value in seconds),
                    "min": expected_bytes / (2**30) / max(seconds),
                    "max": expected_bytes / (2**30) / min(seconds),
                },
            }
        )
    return summaries


def generate_model(directory, weight_count, dimension, seed):
    import numpy as np  # noqa: PLC0415
    import onnx  # noqa: PLC0415
    from onnx import TensorProto, helper  # noqa: PLC0415

    if dimension % 32:
        raise ValueError("--weight-dim must be divisible by 32 for 4-KiB-aligned lengths")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / name for name in ("model.onnx", "weights.bin", "inputs.npz", "expected.npz")]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; use --model to reuse an existing fixture")
    rng = np.random.default_rng(seed)
    initializers, nodes, outputs, expected = [], [], [], {}
    with paths[1].open("wb") as weights_file:
        for index in range(weight_count):
            name, output = f"weight_{index}", f"output_{index}"
            # Binary fractions and a ones input yield stable sums without TF32 ambiguity.
            weight = rng.integers(-8, 9, size=(dimension, dimension), dtype=np.int32).astype("<f4") / 16
            offset = weights_file.tell()
            weights_file.write(weight.tobytes())
            tensor = TensorProto(name=name, data_type=TensorProto.FLOAT, dims=[dimension, dimension])
            tensor.data_location = TensorProto.EXTERNAL
            for key, value in {"location": "weights.bin", "offset": offset, "length": weight.nbytes}.items():
                tensor.external_data.add(key=key, value=str(value))
            initializers.append(tensor)
            nodes.append(helper.make_node("MatMul", ["input", name], [output]))
            outputs.append(helper.make_tensor_value_info(output, TensorProto.FLOAT, [1, dimension]))
            expected[output] = weight.sum(axis=0, dtype=np.float64).astype(np.float32).reshape(1, dimension)
    graph = helper.make_graph(
        nodes,
        "cuda_external_weight_loading",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, dimension])],
        outputs,
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save_model(model, paths[0])
    np.savez(paths[2], input=np.ones((1, dimension), dtype=np.float32))
    np.savez(paths[3], **expected)
    return tuple(str(path) for path in (paths[0], paths[2], paths[3]))


def model_metadata(model_path):
    import onnx  # noqa: PLC0415

    model = onnx.load(model_path, load_external_data=False)
    tensors = list(model.graph.initializer)
    # Count nested graph initializers too; sparse weights are not part of this benchmark.
    graphs = [model.graph]
    while graphs:
        graph = graphs.pop()
        if graph.sparse_initializer:
            raise ValueError("Sparse initializers are not supported by this benchmark")
        for node in graph.node:
            for attribute in node.attribute:
                nested = [attribute.g] if attribute.HasField("g") else list(attribute.graphs)
                for child in nested:
                    tensors.extend(child.initializer)
                    graphs.append(child)
    files, sizes, aligned = set(), [], True
    for tensor in tensors:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        entries = {item.key: item.value for item in tensor.external_data}
        size = math.prod(tensor.dims) * onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).itemsize
        if "length" in entries and int(entries["length"]) != size:
            raise ValueError(f"Unexpected external-data length for {tensor.name}")
        offset = int(entries.get("offset", 0))
        file = (Path(model_path).parent / entries["location"]).resolve()
        if offset < 0 or file.stat().st_size < offset + size:
            raise ValueError(f"External-data range is outside {file}: {tensor.name}")
        files.add(str(file))
        sizes.append(size)
        aligned = aligned and offset % 4096 == 0 and size % 4096 == 0
    if not sizes:
        raise ValueError("The benchmark requires external initializers")
    return {
        "external_weight_bytes": sum(sizes),
        "external_tensor_count": len(sizes),
        "offsets_and_lengths_4096_aligned": aligned,
        "external_files": [{"path": file, "size_bytes": Path(file).stat().st_size} for file in sorted(files)],
    }


def verify_outputs(actual, expected, rtol, atol):
    import numpy as np  # noqa: PLC0415

    if set(actual) != set(expected):
        raise ValueError("Reference NPZ must contain exactly the model's output names")
    for name, reference in expected.items():
        if actual[name].shape != reference.shape or actual[name].dtype != reference.dtype:
            raise ValueError(f"Output shape/dtype differs from reference: {name}")
        if not np.isfinite(reference).all() or not np.isfinite(actual[name]).all():
            raise ValueError(f"Non-finite output/reference: {name}")
        np.testing.assert_allclose(actual[name], reference, rtol=rtol, atol=atol, err_msg=name)


def run_worker(args):
    import numpy as np  # noqa: PLC0415

    import onnxruntime as ort  # noqa: PLC0415

    # Framework mmap/pageable instrumentation uses the default logger, not the session logger.
    ort.set_default_logger_severity(1)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("This ONNX Runtime package does not provide CUDAExecutionProvider")
    with np.load(args.inputs, allow_pickle=False) as archive:
        inputs = dict(archive)
    with np.load(args.expected_outputs, allow_pickle=False) as archive:
        expected = dict(archive)
    if args.evict_file_cache:
        for path in (args.model, *args.external_data):
            evict_file_cache(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.log_severity_level = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry("session.disable_prepacking", "1")
    # Avoid folding/prepacking weights into a different graph or representation.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cuda_options = provider_options(args.path, args.device_id, args.reading_threads, sys.platform)
    start = time.perf_counter()
    session = ort.InferenceSession(
        args.model, sess_options=options, providers=[("CUDAExecutionProvider", cuda_options)]
    )
    elapsed = time.perf_counter() - start
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDAExecutionProvider was requested but is not active")
    session.disable_fallback()
    output_names = [output.name for output in session.get_outputs()]
    actual = dict(zip(output_names, session.run(output_names, inputs), strict=True))
    verify_outputs(actual, expected, args.rtol, args.atol)
    print(
        RESULT_PREFIX
        + json.dumps(
            {
                "seconds": elapsed,
                "correctness": "passed",
                "active_providers": session.get_providers(),
                "provider_options": cuda_options,
                "ort_version": ort.__version__,
                "ort_module": ort.__file__,
                "ort_build_info": ort.get_build_info(),
                "pid": os.getpid(),
            },
            sort_keys=True,
        )
    )


def worker_command(args, path):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--path", path]
    for name in ("model", "inputs", "expected_outputs", "device_id", "threads", "reading_threads", "rtol", "atol"):
        command.extend([f"--{name.replace('_', '-')}", str(getattr(args, name))])
    for file in args.external_data:
        command.extend(["--external-data", file])
    if args.evict_file_cache:
        command.append("--evict-file-cache")
    return command


def run_sample(args, path, expected_bytes):
    process = subprocess.run(worker_command(args, path), capture_output=True, text=True, check=False)
    if process.returncode:
        raise RuntimeError(f"{path} worker failed ({process.returncode}):\n{process.stdout}\n{process.stderr}")
    results = [line[len(RESULT_PREFIX) :] for line in process.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    if len(results) != 1:
        raise RuntimeError(f"Expected one worker result, received {len(results)}:\n{process.stdout}\n{process.stderr}")
    sample = json.loads(results[0])
    observed, warnings = parse_loader_logs(process.stderr)
    sample.update(
        requested_path=path,
        expected_path=direct_path(sys.platform) if path == "direct" else path,
        observed_bytes=observed,
        fallback_warnings=warnings,
        # Keep diagnostic evidence even if this ORT build does not emit route logs.
        stderr=process.stderr,
    )
    sample["status"] = classify_path(sample["expected_path"], observed, expected_bytes)
    return sample


def gpu_metadata():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {"nvidia_smi": None}
    return {"nvidia_smi": result.stdout.strip(), "nvidia_smi_error": result.stderr.strip()}


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Existing ONNX model; requires input and reference NPZ files")
    source.add_argument("--generate-model", metavar="DIRECTORY", help="Create an aligned deterministic MatMul fixture")
    parser.add_argument("--inputs", help="Input NPZ keyed by model input name")
    parser.add_argument("--expected-outputs", help="Independent reference NPZ keyed by model output name")
    parser.add_argument("--external-data", action="append", default=[], help="Additional file to advise eviction")
    parser.add_argument("--weight-count", type=positive_int, default=16, help="Generated external tensor count")
    parser.add_argument(
        "--weight-dim", type=positive_int, default=1024, help="Square weight dimension, divisible by 32"
    )
    parser.add_argument("--seed", type=nonnegative_int, default=2026, help="Fixture and per-round path-order seed")
    parser.add_argument(
        "--device-id", type=nonnegative_int, default=0, help="CUDA logical device ID, shared by all paths"
    )
    parser.add_argument("--threads", type=positive_int, default=96, help="Intra-op thread count")
    parser.add_argument(
        "--reading-threads", type=reading_thread_count, default=4, help="Pinned/direct reading threads, 1-64"
    )
    parser.add_argument("--repetitions", type=positive_int, default=5, help="Measured fresh processes per path")
    parser.add_argument("--warmup", type=nonnegative_int, default=1, help="Unmeasured fresh processes per path")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--evict-file-cache", action="store_true", help="POSIX per-file eviction hint, NOT cold-cache proof"
    )
    parser.add_argument("--output", help="Write the complete JSON report here as well as stdout")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--path", choices=PATHS, help=argparse.SUPPRESS)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    if args.reading_threads == 0:
        parser.error("--reading-threads must be positive; the pageable path always uses 0")
    if any(not math.isfinite(value) or value < 0 for value in (args.rtol, args.atol)):
        parser.error("--rtol and --atol must be finite and nonnegative")
    if args.generate_model:
        args.model, args.inputs, args.expected_outputs = generate_model(
            args.generate_model, args.weight_count, args.weight_dim, args.seed
        )
    if not args.inputs or not args.expected_outputs:
        parser.error("--model requires --inputs and --expected-outputs for correctness verification")
    for name in ("model", "inputs", "expected_outputs"):
        path = Path(getattr(args, name)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, str(path))
    if args.worker:
        if not args.path:
            parser.error("--worker requires --path")
        run_worker(args)
        return
    metadata = model_metadata(args.model)
    args.external_data = sorted({*args.external_data, *(item["path"] for item in metadata["external_files"])})
    for path in args.external_data:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    paths = list(PATHS) if direct_path(sys.platform) else ["pageable", "pinned"]
    report = {
        "metadata": {
            **metadata,
            **gpu_metadata(),
            "model": args.model,
            "device_id": args.device_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "threads": args.threads,
            "reading_threads": args.reading_threads,
            "seed": args.seed,
            "warmup_per_path": args.warmup,
            "repetitions_per_path": args.repetitions,
            "cache_policy": "advisory-eviction-not-guaranteed-cold"
            if args.evict_file_cache
            else "OS-managed; may be warm",
            "timed_region": "InferenceSession construction including weight transfer completion",
            "verification": "blocking CUDA inference and independent NPZ comparison outside timed region",
            "graph_optimization": "disabled",
            "prepacking": "disabled",
            "direct_api": direct_path(sys.platform),
            "direct_status": "requested" if direct_path(sys.platform) else "unavailable on this platform",
            "direct_api_caveat": PATH_DESCRIPTIONS.get(direct_path(sys.platform), "Unavailable on this platform"),
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "warmups": [],
        "samples": [],
    }
    order = random.Random(args.seed)
    for index in range(args.warmup + args.repetitions):
        order.shuffle(paths)
        for path in paths:
            sample = run_sample(args, path, metadata["external_weight_bytes"])
            sample["round"] = index - args.warmup
            report["warmups" if index < args.warmup else "samples"].append(sample)
            print(f"{path}: {sample['status']}, {sample['seconds']:.6f} s", file=sys.stderr)
    report["summary"] = summarize_samples(report["samples"], metadata["external_weight_bytes"])
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
