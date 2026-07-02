# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Profiling script for the CUDA MatMulNBits (4-bit / 8-bit weight-only) op and its online small-M
tuning option (session config ep.cuda.matmulnbits_tune_small_m).

Three subcommands:

  case     Time one K x N weight at a single M. NVTX-wrapped so nsys can capture kernel-level timing.
  latency  Latency table (avg us) across the representative decoder shapes for one config.
  sweep    Does tuning help? For every bits x dtype x block_size x zero-point combination, time each
           shape with the tuner off and on and report the best speedup (built-in / tuned).

The representative shapes are taken from real quantized decoders: Qwen3-4B (q4b:*), Qwen3-8B (q8b:*),
and a Gemma4 target (gm:*).

Usage:
  # Does the online tuning help on this GPU, across every combination (the main question):
  python profile_matmul_nbits.py sweep

  # Narrow the sweep (e.g. only 8-bit, block 32/128):
  python profile_matmul_nbits.py sweep --bits 8 --blocks 32 128

  # Latency table for one config, with the tuner on:
  python profile_matmul_nbits.py latency --bits 4 --dtype bf16 --tune

  # Single case, kernel-level via nsys + the repo parser:
  nsys profile -t cuda,nvtx -o mnb --export=sqlite \
      python profile_matmul_nbits.py case --k 4096 --n 4096 --m 8
  python parse_nsys.py mnb.sqlite --nvtx-range benchmark --pattern '%'
"""

import argparse
import json
import time
from contextlib import nullcontext

import numpy as np
import torch
from onnx import TensorProto, helper

import onnxruntime

try:
    import ml_dtypes

    bfloat16 = ml_dtypes.bfloat16
except ImportError:
    bfloat16 = None

try:
    import nvtx

    has_nvtx = True
except ImportError:
    has_nvtx = False
    nvtx = None

RESULT_PREFIX = "MATMUL_NBITS_RESULT "

_OT = {"fp16": TensorProto.FLOAT16, "bf16": TensorProto.BFLOAT16}
_TT = {"fp16": torch.float16, "bf16": torch.bfloat16}
_ELEM = {torch.float16: TensorProto.FLOAT16, torch.bfloat16: TensorProto.BFLOAT16}

# Representative decoder weight matrices (K = in features, N = out features), taken from real
# quantized decoders: Qwen3-4B (hidden 2560), Qwen3-8B (hidden 4096), and a Gemma4 target (hidden
# 1536, wide MoE). Labels are model:projection. lm_head shapes are large-N and take the chunked
# dequant path (not the small-M batched kernel), so they are omitted from the default small-M set.
DEFAULT_CASES = [
    # Qwen3-4B
    ("q4b:q", 2560, 4096),
    ("q4b:kv", 2560, 1024),
    ("q4b:o", 4096, 2560),
    ("q4b:gate_up", 2560, 9728),
    ("q4b:down", 9728, 2560),
    # Qwen3-8B
    ("q8b:qo", 4096, 4096),
    ("q8b:kv", 4096, 1024),
    ("q8b:gate_up", 4096, 12288),
    ("q8b:down", 12288, 4096),
    # Gemma4 target
    ("gm:q", 1536, 2048),
    ("gm:kv", 1536, 256),
    ("gm:o", 2048, 1536),
    ("gm:gate_up", 1536, 12288),
    ("gm:down", 12288, 1536),
]
DEFAULT_MS = [1, 2, 4, 8, 16]


def _nvtx_range(name, color="green"):
    if not has_nvtx:
        return nullcontext()
    return nvtx.annotate(name, color=color)


def build_model(k, n, block_size, bits, onnx_dtype, with_zero_point=True):
    rng = np.random.default_rng(0)
    n_blocks = (k + block_size - 1) // block_size
    blob = block_size // (8 // bits)
    b = rng.integers(0, 256, size=(n, n_blocks, blob), dtype=np.uint8)
    scales_f32 = rng.random(n * n_blocks).astype(np.float32) * 0.02 + 0.01
    if onnx_dtype == TensorProto.FLOAT16:
        scales = scales_f32.astype(np.float16)
    elif onnx_dtype == TensorProto.BFLOAT16:
        if bfloat16 is None:
            raise RuntimeError("ml_dtypes is required for bf16 (pip install ml_dtypes)")
        scales = scales_f32.astype(bfloat16)
    else:
        scales = scales_f32
    inits = [
        helper.make_tensor("B", TensorProto.UINT8, list(b.shape), b.tobytes(), raw=True),
        helper.make_tensor("scales", onnx_dtype, list(scales.shape), scales.tobytes(), raw=True),
    ]
    inputs = ["A", "B", "scales"]
    if with_zero_point:
        # 4-bit zero points are packed two per byte; 8-bit zero points are one byte per block.
        zp_count = n * ((n_blocks + 1) // 2) if bits == 4 else n * n_blocks
        zp = rng.integers(0, 256, size=(zp_count,), dtype=np.uint8)
        inits.append(helper.make_tensor("zero_points", TensorProto.UINT8, list(zp.shape), zp.tobytes(), raw=True))
        inputs.append("zero_points")
    node = helper.make_node(
        "MatMulNBits",
        inputs,
        ["Y"],
        domain="com.microsoft",
        K=k,
        N=n,
        bits=bits,
        block_size=block_size,
        accuracy_level=0,
    )
    graph = helper.make_graph(
        [node],
        "mnb",
        [helper.make_tensor_value_info("A", onnx_dtype, ["M", k])],
        [helper.make_tensor_value_info("Y", onnx_dtype, ["M", n])],
        inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("", 17)]
    )
    return model.SerializeToString()


MATMULNBITS_TUNE_SMALL_M_CONFIG = "ep.cuda.matmulnbits_tune_small_m"


def make_session(model, tune=False):
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    if tune:
        so.add_session_config_entry(MATMULNBITS_TUNE_SMALL_M_CONFIG, "1")
    return onnxruntime.InferenceSession(model, so, providers=["CUDAExecutionProvider"])


def run_case(name, m, k, n, block_size, bits, dtype, warmup, repeat, tune=False, zp=True, quiet=False):
    onnx_dtype = _OT[dtype]
    torch_dtype = _TT[dtype]
    sess = make_session(build_model(k, n, block_size, bits, onnx_dtype, with_zero_point=zp), tune=tune)
    a = np.random.default_rng(m).random((m, k)).astype(np.float32) * 0.02 - 0.01
    at = torch.from_numpy(a).to(torch_dtype).cuda().contiguous()
    y = torch.empty((m, n), dtype=torch_dtype, device="cuda")
    io = sess.io_binding()
    io.bind_input("A", "cuda", 0, _ELEM[torch_dtype], list(at.shape), at.data_ptr())
    io.bind_output("Y", "cuda", 0, _ELEM[torch_dtype], list(y.shape), y.data_ptr())

    with _nvtx_range("warmup", "yellow"):
        for _ in range(warmup):
            sess.run_with_iobinding(io)
    torch.cuda.synchronize()

    best = float("inf")
    trials = 10
    with _nvtx_range("benchmark", "green"):
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeat):
                sess.run_with_iobinding(io)
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) / repeat)
    us = best * 1e6
    result = {
        "case": name,
        "m": m,
        "k": k,
        "n": n,
        "block_size": block_size,
        "bits": bits,
        "dtype": dtype,
        "zp": int(zp),
        "tune": int(tune),
        "avg_us": round(us, 2),
    }
    if not quiet:
        print(RESULT_PREFIX + json.dumps(result))
    return result


def print_latency_table(args):
    """Latency table (avg us) across the representative shapes for one config, tuner off or on."""
    zp = not args.no_zp
    rows = {
        name: {
            m: run_case(name, m, k, n, args.block_size, args.bits, args.dtype, args.warmup, args.repeat, args.tune, zp)[
                "avg_us"
            ]
            for m in args.ms
        }
        for name, k, n in DEFAULT_CASES
    }
    state = "tuned" if args.tune else "built-in cap"
    zp_tag = "zp" if zp else "no-zp"
    print(
        f"\n  MatMulNBits {args.bits}-bit block{args.block_size} {args.dtype} {zp_tag} [{state}]  (avg us, lower is better)"
    )
    print("  " + f"{'shape':12s} {'K':>6} {'N':>7} " + " ".join(f"M={m:<5}" for m in args.ms))
    for name, k, n in DEFAULT_CASES:
        print("  " + f"{name:12s} {k:>6} {n:>7} " + " ".join(f"{rows[name][m]:7.1f}" for m in args.ms))


def run_sweep(args):
    """Exhaustive check of whether the online-tuning session option helps. For every combination of
    bits x dtype x block_size x zero-point, each representative shape is timed with the tuner off and on;
    the cell is the best speedup (built-in / tuned) over the probed M values, so >1.00 means tuning is
    faster and ~1.00 means it makes no difference. Lets a user on any GPU see where (if anywhere) the
    per-device crossover differs from the built-in cap."""
    for bits in args.bits:
        structural_max = 8 if bits == 8 else 16
        ms = [m for m in args.ms if 2 <= m <= structural_max]
        for dtype in args.dtypes:
            cols = [(b, zp) for b in args.blocks for zp in (True, False)]
            print(f"\n=== {bits}-bit {dtype}: best built-in/tuned speedup over M={ms}  (>1.00 = tuner helps) ===")
            print(
                "  "
                f"{'shape':12s} {'K':>6} {'N':>7} "
                + " ".join((f"b{b}" + ("+zp" if zp else "")).rjust(8) for b, zp in cols)
            )
            for name, k, n in DEFAULT_CASES:
                cells = []
                for b, zp in cols:
                    best = 1.0
                    for m in ms:
                        off = run_case(name, m, k, n, b, bits, dtype, args.warmup, args.repeat, False, zp, quiet=True)
                        on = run_case(name, m, k, n, b, bits, dtype, args.warmup, args.repeat, True, zp, quiet=True)
                        best = max(best, off["avg_us"] / on["avg_us"])
                    cells.append(f"{best:.2f}x")
                print("  " + f"{name:12s} {k:>6} {n:>7} " + " ".join(c.rjust(8) for c in cells))


def main():
    p = argparse.ArgumentParser(description="Profile CUDA MatMulNBits and its online small-M tuning option.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_timing(sp):
        sp.add_argument("--warmup", type=int, default=25)
        sp.add_argument("--repeat", type=int, default=200)

    c = sub.add_parser("case", help="time one K x N weight at a single M (for nsys or a quick check)")
    c.add_argument("--k", type=int, required=True, help="in features")
    c.add_argument("--n", type=int, required=True, help="out features")
    c.add_argument("--m", type=int, required=True, help="rows")
    c.add_argument("--bits", type=int, default=4, choices=[4, 8])
    c.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    c.add_argument("--block-size", type=int, default=32)
    c.add_argument("--no-zp", action="store_true", help="build without zero points")
    c.add_argument("--tune", action="store_true", help="enable the tuning session option")
    add_timing(c)

    lat = sub.add_parser("latency", help="latency table across the representative shapes for one config")
    lat.add_argument("--bits", type=int, default=4, choices=[4, 8])
    lat.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    lat.add_argument("--block-size", type=int, default=32)
    lat.add_argument("--no-zp", action="store_true", help="build without zero points")
    lat.add_argument("--tune", action="store_true", help="enable the tuning session option")
    lat.add_argument("--ms", type=int, nargs="+", default=DEFAULT_MS)
    add_timing(lat)

    sw = sub.add_parser("sweep", help="does tuning help? speedup across bits x dtype x block x zero-point")
    sw.add_argument("--bits", type=int, nargs="+", default=[4, 8], choices=[4, 8])
    sw.add_argument("--dtypes", nargs="+", default=["fp16", "bf16"], choices=["fp16", "bf16"])
    sw.add_argument("--blocks", type=int, nargs="+", default=[16, 32, 64, 128])
    sw.add_argument("--ms", type=int, nargs="+", default=[6, 16], help="row counts to probe for the best speedup")
    add_timing(sw)

    args = p.parse_args()
    if args.cmd == "case":
        run_case(
            "case",
            args.m,
            args.k,
            args.n,
            args.block_size,
            args.bits,
            args.dtype,
            args.warmup,
            args.repeat,
            args.tune,
            not args.no_zp,
        )
    elif args.cmd == "latency":
        print_latency_table(args)
    elif args.cmd == "sweep":
        run_sweep(args)


if __name__ == "__main__":
    main()
