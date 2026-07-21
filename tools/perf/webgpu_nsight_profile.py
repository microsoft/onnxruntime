# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#!/usr/bin/env python3

"""WebGPU EP Nsight Graphics profiling helper.

Runs a single-op WebGPU workload under Nsight Graphics's headless
`ngfx.exe` GPU Trace Profiler activity and produces a `.ngfx-gputrace`
report with per-dispatch hardware counters (SM occupancy, warp stalls by
reason, L1TEX / L2 hit rates, DRAM throughput, tensor pipe %, ...).

Why Nsight Graphics and not Nsight Compute (`ncu`) or Nsight Systems (`nsys`)?

  - `ncu` hooks the CUDA driver (`cuLaunchKernel`). WebGPU dispatches go
    through D3D12 / Vulkan and never touch the CUDA driver, so `ncu`
    reports "No kernels were profiled." This is a design limitation of
    `ncu`, not a build flag we can flip.
  - `nsys` traces D3D12 / Vulkan just fine and gives you a timeline, but
    its GPU metrics are periodically sampled at ~10 kHz — coarse SM %,
    DRAM BW, PCIe throughput. It cannot give you per-dispatch warp stall
    reasons, per-dispatch SM occupancy, or per-dispatch cache hit rates.
  - **Nsight Graphics GPU Trace** hooks D3D12 / Vulkan directly. Under
    the hood it uses the same NvPerf counter backend as `ncu` and
    produces the same class of per-dispatch counter data — but for
    graphics-API compute shaders instead of CUDA kernels.

See docs/WebGPU-EP-Nsight-Graphics-Profiling.md for the full "why ncu is
wrong for this" discussion and the graphics-API-vs-CUDA metric mapping.

Requires:
  - Nsight Graphics 2024.x or newer installed (Windows or Linux).
    Download: https://developer.nvidia.com/nsight-graphics
    The script auto-discovers `ngfx.exe` under the default install path;
    override with the `NGFX` env var.
  - **Elevated permissions.** NVIDIA drivers restrict GPU performance
    counter access to admins by default. Either run this script from an
    elevated shell, or enable "Allow access to the GPU performance
    counters to all users" in NVIDIA Control Panel → Developer → Manage
    GPU Performance Counters. See
    https://developer.nvidia.com/ERR_NVGPUCTRPERM for details.
  - An ORT build with `onnxruntime_ENABLE_NSIGHT_FOR_WEBGPU_EP=ON`.
    The workload session always sets `enableNsightProfiling=1` on the
    WebGPU EP so each dispatch is in its own compute pass with a
    `PushDebugGroup("op=<op>|node=<name>")` label around it. Nsight
    Graphics's GPU Trace filter picks up those labels natively.

Sample invocations:

  # Plain MatMul (M, N, K, dtype required)
  python webgpu_nsight_profile.py \\
      --op MatMul --shape "M=1,N=8192,K=3072,dtype=fp16"

  # MatMulNBits (M, N, K, bits, block_size, dtype required)
  python webgpu_nsight_profile.py \\
      --op MatMulNBits \\
      --shape "M=1,N=8192,K=3072,bits=4,block_size=32,dtype=fp16"

  # MatMulNBitsMlp (Qwen-style fused gate/up + RMSNorm)
  python webgpu_nsight_profile.py \\
      --op MatMulNBitsMlp \\
      --shape "M=1,N=8192,K=3072,bits=4,block_size=32,dtype=fp16,gate_activation=silu,rms_norm=1,skip=0"

Flow:

  1. The script allocates a per-run output directory under
     `$ORT_PROFILE_OUT` (default `~/perf_runs/<date>/<op>/<time>/`).
  2. It spawns `ngfx.exe --activity "GPU Trace Profiler"` pointing at
     this same script as the workload target (via a hidden `--_workload`
     flag). The workload builds a one-node ONNX for the requested
     op/shape, opens an `InferenceSession` on the WebGPU EP with
     `enableNsightProfiling=1`, and loops `session.run()` under Nsight
     Graphics's injection shim.
  3. Nsight Graphics waits `--warmup-submits` GPU submits, then captures
     the next `--capture-submits` (capped by `--capture-timeout-ms`),
     collects all per-dispatch counters, and writes a
     `<something>.ngfx-gputrace` report to the output directory.
  4. When `--auto-export` fires, per-dispatch tables are also written as
     Excel files (GPUTRACE_FRAME.xls, GPUTRACE_REGIMES.xls,
     D3DPERF_EVENTS.xls, ...) for quick grep / diffing without opening
     the UI.
  5. The workload subprocess is terminated automatically.

Open the `.ngfx-gputrace` in Nsight Graphics's UI for the full
Range Profiler view; filter by debug group `op=<op>` to isolate a
specific op's dispatches.

Notes:

  - Backend defaults to the platform default (D3D12 on Windows, Vulkan
    on Linux). Windows callers can override with `--backend vulkan`; on
    Linux only Vulkan is supported by Dawn.
  - Output directory contains: `<op>.onnx`, `metadata.json`,
    `wgsl_dump.wgsl`, `*.ngfx-gputrace`, and (with `--auto-export`) a
    `BASE/` subdirectory of Excel tables.

See docs/WebGPU-EP-Nsight-Graphics-Profiling.md for the full design and
usage guide.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
# Common helpers
# ============================================================================


def parse_shape(shape_str: str) -> dict[str, object]:
    """Parse "M=1,N=8192,K=3072,bits=4,block=32,dtype=fp16" into a dict.

    Values that look like ints are coerced to int; everything else stays str.
    """
    d: dict[str, object] = {}
    for kv in shape_str.split(","):
        k, _, v = kv.partition("=")
        k, v = k.strip(), v.strip()
        if not k or not v:
            continue
        try:
            d[k] = int(v)
        except ValueError:
            d[k] = v
    return d


def _require(shape: dict, op: str, keys: list[str]) -> None:
    """Fail fast when required shape keys are missing.

    Profiling harnesses that silently default shape parameters (M, bits, block_size, ...)
    are footguns: a wrong assumption produces a valid-looking capture that measures the
    wrong kernel geometry. Force the caller to spell everything out.
    """
    missing = [k for k in keys if k not in shape]
    if missing:
        sys.exit(
            f"{op}: required shape parameter(s) missing: {missing}. "
            f"Required keys for {op}: {keys}. Got: {sorted(shape)}."
        )


def find_ngfx() -> Path:
    """Locate the `ngfx.exe` (Windows) / `ngfx` (Linux) headless CLI.

    Preference order:
      1. `NGFX` env var if set.
      2. `ngfx` / `ngfx.exe` on PATH.
      3. Default Windows install path
         `C:\\Program Files\\NVIDIA Corporation\\Nsight Graphics
         <version>\\host\\windows-desktop-nomad-x64\\ngfx.exe`; if
         multiple versions are installed, the lexicographically latest
         wins.

    Exits with a clear message if not found.
    """
    ngfx = os.environ.get("NGFX") or shutil.which("ngfx") or shutil.which("ngfx.exe")
    if not ngfx and platform.system() == "Windows":
        root = Path(r"C:\Program Files\NVIDIA Corporation")
        if root.exists():
            candidates = sorted(
                root.glob("Nsight Graphics*/host/windows-desktop-nomad-x64/ngfx.exe"),
                reverse=True,
            )
            if candidates:
                ngfx = str(candidates[0])
    if not ngfx:
        sys.exit(
            "ngfx not found. Install NVIDIA Nsight Graphics and either:\n"
            "  - set NGFX=/full/path/to/ngfx(.exe), or\n"
            "  - put ngfx on PATH.\n"
            "Download: https://developer.nvidia.com/nsight-graphics."
        )
    p = Path(ngfx)
    if not p.exists():
        sys.exit(f"NGFX={ngfx!r} does not exist.")
    return p


def check_nvidia_gpu_present() -> None:
    """Warn (best-effort) if this box has no NVIDIA GPU.

    Nsight Graphics can't collect SM counters on AMD / Intel / Apple GPUs;
    it will report the error itself when you attach, but a heads-up here
    saves a round-trip through the GUI. Best-effort detection via the OS's
    GPU enumeration; on failure we stay silent.
    """
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name"],
                capture_output=True, text=True, timeout=5,
            )
            gpus = result.stdout
        else:  # Linux (macOS falls through with empty output)
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5,
            )
            gpus = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return  # can't check; stay silent

    if not gpus:
        return
    if "NVIDIA" in gpus.upper():
        return

    print(
        "WARNING: No NVIDIA GPU detected on this machine. Nsight Graphics's SM\n"
        "counter path (warp stalls, occupancy, cache hit rates) is NVIDIA-only.\n"
        f"Detected GPUs:\n{gpus.strip()}\n\n"
        "On non-NVIDIA hardware, use the vendor's own profiler instead:\n"
        "  - Radeon GPU Profiler (RGP) on AMD Vulkan,\n"
        "  - Intel GPA on Intel,\n"
        "  - Xcode Instruments Metal System Trace on Apple silicon.\n"
        "The workload will still launch; feel free to attach a non-Nsight profiler.\n"
        "Bypass this warning with ORT_WEBGPU_PROFILE_SKIP_GPU_CHECK=1.\n",
        file=sys.stderr,
    )


def output_dir(op: str) -> Path:
    """Return a deterministic per-day/per-op/per-timestamp directory. Creates it."""
    root_str = os.environ.get("ORT_PROFILE_OUT")
    root = Path(root_str) if root_str else Path.home() / "perf_runs"
    day = datetime.now().strftime("%Y-%m-%d")
    time_ = datetime.now().strftime("%H%M%S")
    out = root / day / op / time_
    out.mkdir(parents=True, exist_ok=True)
    return out


def default_backend() -> str:
    """Match the WebGPU EP's own default backend for this platform.

    Windows: D3D12 (production ORT default; matches what users actually ship).
    Linux / others: Vulkan (only backend Dawn ships there).
    """
    return "d3d12" if platform.system() == "Windows" else "vulkan"


def valid_backends() -> list[str]:
    """Backends the EP can be built against for this platform.

    Windows: both D3D12 and Vulkan (Dawn supports both).
    Linux: Vulkan only.
    """
    return ["d3d12", "vulkan"] if platform.system() == "Windows" else ["vulkan"]


# ============================================================================
# ONNX model builders
#
# One builder per op. Adding a new op = add a builder + register it in OP_BUILDERS.
# onnx / numpy are imported inside the builders so the parent invocation (which
# just prints attach instructions) can run without them installed — only the
# workload child actually needs them.
# ============================================================================


def _build_matmul_nbits(shape: dict) -> "onnx.ModelProto":
    """Build a single-node com.microsoft.MatMulNBits model.

    Required shape parameters: M, N, K, bits, block_size, dtype.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    _require(shape, "MatMulNBits", ["M", "N", "K", "bits", "block_size", "dtype"])

    M = int(shape["M"])
    N = int(shape["N"])
    K = int(shape["K"])
    bits = int(shape["bits"])
    block_size = int(shape["block_size"])
    dtype_str = str(shape["dtype"]).lower()
    if dtype_str not in ("fp16", "float16", "fp32", "float32"):
        sys.exit(f"dtype must be fp16 or fp32 (got {dtype_str!r}).")
    ort_dtype = TensorProto.FLOAT16 if dtype_str in ("fp16", "float16") else TensorProto.FLOAT

    if K % block_size != 0:
        sys.exit(f"K ({K}) must be a multiple of block_size ({block_size}).")
    if bits not in (2, 4, 8):
        sys.exit(f"bits must be 2, 4, or 8 (got {bits}).")

    a = helper.make_tensor_value_info("A", ort_dtype, [1, M, K])
    y = helper.make_tensor_value_info("Y", ort_dtype, [1, M, N])

    packed_bytes_per_block = block_size * bits // 8
    n_blocks_per_col = K // block_size
    rng = np.random.default_rng(0xC0FFEE)
    B_bytes = rng.integers(
        0, 255, size=(N, n_blocks_per_col, packed_bytes_per_block), dtype=np.uint8
    ).tobytes()
    B_init = helper.make_tensor(
        "B",
        TensorProto.UINT8,
        [N, n_blocks_per_col, packed_bytes_per_block],
        B_bytes,
        raw=True,
    )

    scales_np_dtype = np.float16 if ort_dtype == TensorProto.FLOAT16 else np.float32
    scales = (rng.standard_normal((N, n_blocks_per_col)).astype(scales_np_dtype) * 0.01)
    scales_init = helper.make_tensor(
        "scales", ort_dtype, [N, n_blocks_per_col], scales.tobytes(), raw=True
    )

    node = helper.make_node(
        "MatMulNBits",
        inputs=["A", "B", "scales"],
        outputs=["Y"],
        domain="com.microsoft",
        K=K,
        N=N,
        block_size=block_size,
        bits=bits,
    )
    graph = helper.make_graph([node], "matmul_nbits_single", [a], [y], [B_init, scales_init])
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("com.microsoft", 1),
            helper.make_opsetid("", 21),
        ],
    )


def _build_matmul(shape: dict) -> "onnx.ModelProto":
    """Build a single-node standard MatMul model.

    Required shape parameters: M, N, K, dtype.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    _require(shape, "MatMul", ["M", "N", "K", "dtype"])

    M = int(shape["M"])
    N = int(shape["N"])
    K = int(shape["K"])
    dtype_str = str(shape["dtype"]).lower()
    if dtype_str not in ("fp16", "float16", "fp32", "float32"):
        sys.exit(f"dtype must be fp16 or fp32 (got {dtype_str!r}).")
    ort_dtype = TensorProto.FLOAT16 if dtype_str in ("fp16", "float16") else TensorProto.FLOAT
    np_dtype = np.float16 if ort_dtype == TensorProto.FLOAT16 else np.float32

    a = helper.make_tensor_value_info("A", ort_dtype, [1, M, K])
    y = helper.make_tensor_value_info("Y", ort_dtype, [1, M, N])

    B_init = helper.make_tensor(
        "B",
        ort_dtype,
        [K, N],
        np.random.default_rng(0xBEEF).standard_normal((K, N)).astype(np_dtype).tobytes(),
        raw=True,
    )

    node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["Y"])
    graph = helper.make_graph([node], "matmul_single", [a], [y], [B_init])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def _make_packed_uint8_weight(
    name: str, N: int, K: int, bits: int, block_size: int, seed: int
) -> "onnx.TensorProto":
    """Build a MatMulNBits-style packed uint8 weight initializer."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    packed_bytes_per_block = block_size * bits // 8
    n_blocks_per_col = K // block_size
    rng = np.random.default_rng(seed)
    data = rng.integers(
        0, 255, size=(N, n_blocks_per_col, packed_bytes_per_block), dtype=np.uint8
    ).tobytes()
    return helper.make_tensor(
        name, TensorProto.UINT8, [N, n_blocks_per_col, packed_bytes_per_block], data, raw=True
    )


def _make_scales(
    name: str, N: int, K: int, block_size: int, ort_dtype: int, seed: int
) -> "onnx.TensorProto":
    """Build a per-block scales initializer with shape [N, K/block_size]."""
    import numpy as np
    from onnx import TensorProto, helper

    np_dtype = np.float16 if ort_dtype == TensorProto.FLOAT16 else np.float32
    n_blocks_per_col = K // block_size
    rng = np.random.default_rng(seed)
    arr = (rng.standard_normal((N, n_blocks_per_col)) * 0.01).astype(np_dtype)
    return helper.make_tensor(name, ort_dtype, [N, n_blocks_per_col], arr.tobytes(), raw=True)


def _build_matmul_nbits_mlp(shape: dict) -> "onnx.ModelProto":
    """Build a single-node com.microsoft.MatMulNBitsMlp model.

    This is the fused Qwen-style gate/up MLP kernel:
        gate = MatMulNBits(A, gate_B, gate_scales)
        up   = MatMulNBits(A, up_B,   up_scales)
        Y    = gate_activation(gate) * up

    Required shape parameters: M, N, K, bits, block_size, dtype, gate_activation,
    rms_norm, skip.

    - gate_activation: currently only "silu" is supported by the WebGPU kernel.
    - rms_norm=0 or 1: fuses SimplifiedLayerNormalization on A (RMSNorm over K).
    - skip=0 or 1: fuses SkipSimplifiedLayerNormalization on A. skip=1 implies
      rms_norm=1 (skip is a residual add fed into the same RMSNorm).

    The MLP schema exposes an `epsilon` attribute for the fused RMSNorm; the ORT
    schema defaults it to 1e-5 and no runtime path exercises other values, so we
    hard-code 1e-5 here rather than making callers thread another knob.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    _require(
        shape,
        "MatMulNBitsMlp",
        ["M", "N", "K", "bits", "block_size", "dtype", "gate_activation", "rms_norm", "skip"],
    )

    M = int(shape["M"])
    N = int(shape["N"])
    K = int(shape["K"])
    bits = int(shape["bits"])
    block_size = int(shape["block_size"])
    dtype_str = str(shape["dtype"]).lower()
    if dtype_str not in ("fp16", "float16", "fp32", "float32"):
        sys.exit(f"dtype must be fp16 or fp32 (got {dtype_str!r}).")
    ort_dtype = TensorProto.FLOAT16 if dtype_str in ("fp16", "float16") else TensorProto.FLOAT
    np_dtype = np.float16 if ort_dtype == TensorProto.FLOAT16 else np.float32
    gate_activation = str(shape["gate_activation"]).lower()
    fuse_rms_norm = int(shape["rms_norm"]) != 0
    fuse_skip = int(shape["skip"]) != 0
    if fuse_skip and not fuse_rms_norm:
        sys.exit(
            "MatMulNBitsMlp: skip=1 requires rms_norm=1 "
            "(SkipSimplifiedLayerNormalization is a superset of SimplifiedLayerNormalization)."
        )
    # Epsilon is fixed by the ORT schema default; no runtime path exercises other values.
    eps = 1e-5

    if K % block_size != 0:
        sys.exit(f"K ({K}) must be a multiple of block_size ({block_size}).")
    if bits != 4:
        sys.exit(f"MatMulNBitsMlp WebGPU kernel currently supports only bits=4 (got {bits}).")
    if block_size != 32:
        sys.exit(f"MatMulNBitsMlp WebGPU kernel currently supports only block_size=32 (got {block_size}).")
    if gate_activation != "silu":
        sys.exit(
            f"MatMulNBitsMlp WebGPU kernel currently supports only "
            f"gate_activation=silu (got {gate_activation})."
        )

    a = helper.make_tensor_value_info("A", ort_dtype, [1, M, K])
    y = helper.make_tensor_value_info("Y", ort_dtype, [1, M, N])

    # Node inputs are positional per the schema; unused optional inputs use "".
    inputs = ["A"]
    initializers = []

    if fuse_skip:
        # skip input tracks the same shape/dtype as A (elementwise residual).
        inputs.append("skip")
        skip_info = helper.make_tensor_value_info("skip", ort_dtype, [1, M, K])
    else:
        inputs.append("")
        skip_info = None

    if fuse_rms_norm:
        inputs.append("norm_scale")
        rng = np.random.default_rng(0xE55EA1)
        norm_arr = rng.standard_normal((K,)).astype(np_dtype)
        initializers.append(
            helper.make_tensor("norm_scale", ort_dtype, [K], norm_arr.tobytes(), raw=True)
        )
    else:
        inputs.append("")

    # gate projection (inputs 3, 4, 5)
    initializers.append(_make_packed_uint8_weight("gate_B", N, K, bits, block_size, seed=0xDEAD01))
    initializers.append(_make_scales("gate_scales", N, K, block_size, ort_dtype, seed=0xDEAD02))
    inputs += ["gate_B", "gate_scales", ""]  # no bias

    # up projection (inputs 6, 7, 8)
    initializers.append(_make_packed_uint8_weight("up_B", N, K, bits, block_size, seed=0xBEEF01))
    initializers.append(_make_scales("up_scales", N, K, block_size, ort_dtype, seed=0xBEEF02))
    inputs += ["up_B", "up_scales", ""]  # no bias

    node = helper.make_node(
        "MatMulNBitsMlp",
        inputs=inputs,
        outputs=["Y"],
        domain="com.microsoft",
        K=K,
        N=N,
        bits=bits,
        block_size=block_size,
        activation=gate_activation,
        epsilon=eps,
    )

    graph_inputs = [a]
    if skip_info is not None:
        graph_inputs.append(skip_info)
    graph = helper.make_graph(
        [node], "matmul_nbits_mlp_single", graph_inputs, [y], initializers
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("com.microsoft", 1),
            helper.make_opsetid("", 21),
        ],
    )


# Register op builders. Keys are lower-case with underscores; the CLI accepts
# either casing. The raw --op string is also used in the debug-group label
# `op=<op>|node=<name>` that Nsight Graphics filters by, so pick a stable name.
OP_BUILDERS = {
    "matmul": _build_matmul,
    "matmulnbits": _build_matmul_nbits,
    "matmul_nbits": _build_matmul_nbits,
    "matmulnbitsmlp": _build_matmul_nbits_mlp,
    "matmul_nbits_mlp": _build_matmul_nbits_mlp,
}


# ============================================================================
# Workload — the child process Nsight Graphics attaches to
# ============================================================================


def _run_workload(args: argparse.Namespace) -> int:
    """Build a single-op model and loop it indefinitely for Nsight Graphics.

    This function is only entered when the parent has re-invoked this script
    via the hidden `--_workload` flag. `enableNsightProfiling=1` is always
    set on the WebGPU EP, so the ORT build must have
    `onnxruntime_ENABLE_NSIGHT_FOR_WEBGPU_EP=ON`.

    The loop is unbounded — Nsight Graphics needs a live process to attach
    to and capture from, and the user decides when to stop by pressing
    Ctrl+C in the parent (which forwards SIGINT here).
    """
    import numpy as np
    import onnx
    import onnxruntime as ort

    op_key = args.op.lower().replace("-", "_")
    builder = OP_BUILDERS.get(op_key)
    if builder is None:
        sys.exit(f"Unknown op {args.op!r}. Supported: {sorted(set(OP_BUILDERS))}")

    shape = parse_shape(args.shape)
    model = builder(shape)

    # The parent process puts the ONNX destination in the env so the output
    # directory is self-describing. `_run_profile` sets
    # ORT_WEBGPU_NSIGHT_ONNX_PATH to `<out_dir>/<op>.onnx` before spawning
    # this child; the child just writes there.
    dump_env = os.environ.get("ORT_WEBGPU_NSIGHT_ONNX_PATH")
    if not dump_env:
        sys.exit(
            "ORT_WEBGPU_NSIGHT_ONNX_PATH not set. The workload subprocess is only "
            "meant to be launched by this script's parent invocation, which sets "
            "the ONNX output path in the environment."
        )
    dump_path = Path(dump_env).expanduser().resolve()
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dump_path))
    print(f"[workload] model: {dump_path}", flush=True)
    # Diagnostic — makes it obvious which onnxruntime install is loaded.
    # If this doesn't point at a build with ENABLE_NSIGHT_FOR_WEBGPU_EP=ON,
    # the capture will have no `op=<op>` debug groups.
    print(f"[workload] onnxruntime: {Path(ort.__file__).parent}", flush=True)

    so = ort.SessionOptions()
    provider_opts = {"backend_type": args.backend, "enableNsightProfiling": "1"}
    providers = [("WebGpuExecutionProvider", provider_opts), "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(dump_path), so, providers=providers)

    # Build deterministic random inputs from the model's declared input tensors.
    ort_type_to_np = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    rng = np.random.default_rng(0x0DDBA11)
    ort_inputs: dict[str, np.ndarray] = {}
    for meta in sess.get_inputs():
        np_dtype = ort_type_to_np.get(meta.type)
        if np_dtype is None:
            sys.exit(f"Unsupported input dtype in model: {meta.type}")
        dims = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
        if np.issubdtype(np_dtype, np.floating):
            arr = rng.standard_normal(dims).astype(np_dtype)
        elif np_dtype is np.bool_:
            arr = rng.integers(0, 2, size=dims).astype(np.bool_)
        else:
            arr = rng.integers(0, 10, size=dims).astype(np_dtype)
        ort_inputs[meta.name] = arr

    # Emit a marker line the parent can grep for to know the workload is
    # ready — useful once we add Pattern B (headless ngfx-cli capture) that
    # needs to know when to trigger the capture.
    print("[workload] READY", flush=True)

    try:
        while True:
            sess.run(None, ort_inputs)
    except KeyboardInterrupt:
        print("[workload] stopping (Ctrl+C received)", flush=True)
    return 0


# ============================================================================
# Profile — spawn the workload and print Nsight Graphics attach instructions
# ============================================================================


def _run_profile(args: argparse.Namespace) -> int:
    """Spawn the workload child and print Nsight Graphics attach instructions.

    Nsight Graphics is GUI-driven: there is no equivalent of `ncu ... -- python
    workload.py` that produces a report file. Instead the user opens the
    Nsight Graphics UI, chooses "Start Activity → GPU Trace" (or "Attach"),
    and points it at either a target executable or a running process.

    This function does the setup that would otherwise be manual:
      - allocates a per-run output directory
      - saves the generated ONNX and (via env) the WGSL dump alongside it
      - spawns the workload subprocess with `enableNsightProfiling=1` and a
        long-running loop
      - prints the exact "Application Executable" / "Working Directory" /
        "Command Line Arguments" to paste into the Nsight Graphics dialog
      - blocks until the user Ctrl+Cs; forwards the interrupt to the child
    """
    if os.environ.get("ORT_WEBGPU_PROFILE_SKIP_GPU_CHECK") != "1":
        check_nvidia_gpu_present()

    out = output_dir(args.op)
    dump_model_path = out / f"{args.op}.onnx"

    env = os.environ.copy()
    # Dump generated WGSL alongside the ONNX so the Shader source view in
    # Nsight Graphics can be correlated with the original WGSL. Respect any
    # pre-existing value.
    env.setdefault("ORT_WEBGPU_EP_SHADER_DUMP_FILE", str(out / "wgsl_dump.wgsl"))
    # Tell the workload child where to save the generated ONNX. Threaded via
    # env so the CLI stays free of internal-only flags.
    env["ORT_WEBGPU_NSIGHT_ONNX_PATH"] = str(dump_model_path)

    workload_cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--_workload",
        "--op", args.op,
        "--shape", args.shape,
        "--backend", args.backend,
    ]

    # Write metadata before running so it survives a crash / Ctrl+C.
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "op": args.op,
                "shape": args.shape,
                "backend": args.backend,
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "workload_cmd": workload_cmd,
                "workload_env_overrides": {
                    "ORT_WEBGPU_EP_SHADER_DUMP_FILE":
                        env["ORT_WEBGPU_EP_SHADER_DUMP_FILE"],
                    "ORT_WEBGPU_NSIGHT_ONNX_PATH":
                        env["ORT_WEBGPU_NSIGHT_ONNX_PATH"],
                },
            },
            indent=2,
        )
    )

    ngfx = find_nsight_graphics()
    ngfx_hint = (
        f"\nNsight Graphics install: {ngfx}\n"
        f"  Launch its UI (usually named `ngfx-ui.exe` under `{ngfx}` on Windows,\n"
        f"  or `nv-nsight-gfx-ui` on Linux) and follow the attach steps below."
        if ngfx else
        "\nNsight Graphics not found on this machine. Install it from\n"
        "  https://developer.nvidia.com/nsight-graphics\n"
        "  and re-run this script (or set the NSIGHT_GRAPHICS env var to its\n"
        "  install root)."
    )

    # Build the attach-instructions message before spawning so it lands on
    # the user's terminal before ORT init spam from the child.
    args_str = " ".join(
        (f'"{a}"' if " " in a else a) for a in workload_cmd[1:]
    )

    # Suggested max-metrics ngfx.exe CLI. Rationale for each flag:
    #   --activity "GPU Trace Profiler"
    #        Compute/graphics profiling activity (not Frame Debugger).
    #   --nvperf-metric-set "workload_metrics_all"
    #        Enables every metric section (SM occupancy, warp stalls, LSU,
    #        L1TEX, L2, DRAM, tensor pipe, ...). Roughly 10x replay cost
    #        per dispatch vs. the default "standard" set — worth it for a
    #        one-shot deep-dive capture. If your ngfx version rejects this
    #        name, try "full" or "advanced_workload".
    #   --advanced-mode-analysis
    #        Enables Nsight Graphics's Peak-Performance-Percentage (PPP)
    #        analysis that populates the Range Profiler's "Analysis" tab
    #        with a ranked list of "% frame time you'd save by fixing X".
    #        Without this flag the Analysis tab is empty.
    #   --collect-shader-source
    #        Attaches source + per-instruction sample counts to every
    #        dispatch so the Shader Source view is populated.
    #   --sampling
    #        Enables PC sampling; feeds the per-instruction stall attribution
    #        that the Shader Source view uses.
    #   --start-after-submits N / --limit-to-submits M / --max-duration-ms
    #        Skip the first N submits (typically warmup) then capture up to
    #        M consecutive submits (or until the wall-clock cap fires).
    #   --auto-export
    #        Also dump per-dispatch tables as .xls under a BASE/ sibling of
    #        the .ngfx-gputrace so you can grep / pivot without opening the
    #        UI.
    workload_dir = Path(dump_model_path).parent
    ngfx_exe_hint = ngfx if ngfx else (
        r'C:\Program Files\NVIDIA Corporation'
        r'\Nsight Graphics <version>'
        r'\host\windows-desktop-nomad-x64\ngfx.exe'
    )
    ngfx_cli = (
        f'"{ngfx_exe_hint}" '
        f'--activity "GPU Trace Profiler" '
        f'--exe "{sys.executable}" '
        f'--args "{args_str}" '
        f'--working-dir "{Path.cwd()}" '
        f'--project-directory "{workload_dir}" '
        f'--start-after-submits 3 '
        f'--limit-to-submits 20 '
        f'--max-duration-ms 30000 '
        f'--nvperf-metric-set "workload_metrics_all" '
        f'--advanced-mode-analysis '
        f'--collect-shader-source '
        f'--sampling '
        f'--auto-export'
    )

    banner = f"""
================================================================================
  WebGPU EP Nsight Graphics profiling — Pattern A (attach)
================================================================================
  Output dir:       {out}
  ONNX (will be):   {dump_model_path}
  WGSL dump:        {env['ORT_WEBGPU_EP_SHADER_DUMP_FILE']}
  Metadata:         {out / 'metadata.json'}
  Backend:          {args.backend}
{ngfx_hint}

  OPTION A — GUI attach (simplest):
    In Nsight Graphics: File → New Activity → GPU Trace (or "Attach" if
    the workload is already running). Paste the following:

      Application Executable  : {sys.executable}
      Working Directory       : {Path.cwd()}
      Command Line Arguments  : {args_str}

    Under "Metrics Set", pick "Workload Metrics (All)" and enable
    "Advanced Mode Analysis" + "Collect Shader Source" for max metrics.

  OPTION B — Headless ngfx.exe (max metrics, from an ADMIN PowerShell):

    {ngfx_cli}

    This collects the full peak-performance-percentage analysis (populates
    the "Analysis" tab), all HW counter sections (SM/LSU/L1TEX/L2/DRAM/
    tensor pipes), shader source with per-instruction sample counts, and
    exports per-dispatch .xls tables next to the .ngfx-gputrace. Cost:
    each target dispatch is replayed ~10x, so wall clock is much longer
    than a normal run — that's expected.

  The workload is being spawned now with `enableNsightProfiling=1` on the
  WebGPU EP, which emits one dispatch per compute pass wrapped in a
  `PushDebugGroup("op={args.op}|node=<name>")` label. Filter GPU Trace's
  Debug Groups panel (or switch the events view to Hierarchical) and search
  for `op={args.op}` to isolate the dispatch of interest.

  When you're done capturing, press Ctrl+C here to stop the workload.
================================================================================
"""
    print(banner, flush=True)

    proc = subprocess.Popen(workload_cmd, env=env)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        # Forward SIGINT to the child so its `except KeyboardInterrupt`
        # branch fires and it prints a clean exit line.
        print("\n[parent] Ctrl+C received; stopping workload...", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 0


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="webgpu_nsight_profile",
        description=(
            "WebGPU EP Nsight Graphics profiling launcher. Builds a single-op ONNX "
            "model, spawns a long-running workload with per-dispatch debug groups, "
            "and prints the exact 'Start Activity → GPU Trace' values to paste into "
            "Nsight Graphics's UI. See docs/WebGPU-EP-Nsight-Graphics-Profiling.md."
        ),
    )
    p.add_argument(
        "--op",
        required=True,
        help=f"Op name. Supported: {sorted(set(OP_BUILDERS))}. Used verbatim in the "
        "PushDebugGroup label (op=<op>|node=<name>) that Nsight Graphics's GPU "
        "Trace filter picks up.",
    )
    p.add_argument(
        "--shape",
        required=True,
        help=(
            "Comma-separated shape / attribute values. Required keys depend on --op; "
            "run with an incomplete --shape to see the required list. Example for "
            'MatMulNBits: "M=1,N=8192,K=3072,bits=4,block_size=32,dtype=fp16".'
        ),
    )
    p.add_argument(
        "--backend",
        choices=valid_backends(),
        default=default_backend(),
        help=(
            "Dawn backend the WebGPU EP should use. Defaults to the platform default "
            "(D3D12 on Windows, Vulkan on Linux). On Windows both are valid; on Linux "
            "only Vulkan is supported by Dawn."
        ),
    )
    # Internal flag used when this script re-invokes itself as the Nsight-attached
    # workload child. Hidden from --help; the leading underscore signals
    # "not for external callers." The ONNX dump path is threaded through the
    # ORT_WEBGPU_NSIGHT_ONNX_PATH env var, so this is the only routing flag.
    p.add_argument("--_workload", action="store_true",
                   dest="workload", help=argparse.SUPPRESS)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.workload:
        return _run_workload(args)
    return _run_profile(args)


if __name__ == "__main__":
    sys.exit(main())
