# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Offline tactic tuning tool for the MatMulNBits fpA_intB CUDA path.

This tool tunes the weight-only GEMM tactics for a model's ``MatMulNBits`` nodes and
writes the results to a persistent tactic cache file that ONNX Runtime can reuse on
later runs (see docs/contrib_ops/cuda/gemm_profiler_cache.md).

How it works:
  * It sets the tuning environment variables and creates a CUDA execution-provider
    session for the model. Kernel construction profiles the configured M buckets and
    writes them to ``<output-prefix>.matmulnbits_fpa_intb.tsv``.
  * It then (best-effort) runs dummy inferences at each requested M value so that any
    additional buckets are profiled lazily and merged into the same cache file.

The generated cache is hardware/build specific: it is only reused on the same GPU
model + SM + CUDA runtime + ORT version/commit/build config.

Example:
    python -m onnxruntime.tools.fpa_intb_tune \
        --model model.onnx \
        --output-prefix /path/to/cache/mymodel \
        --enable-gemv \
        --m-values 1,8,16,32,64,128,256,512,1024,2048
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import onnxruntime as ort

# fpA_intB gemm option bits, mirrored from contrib_ops/cuda/quantization/matmul_nbits.h.
_FPA_INTB_OPTION_ALL = 0x01  # enables both GEMM and the CUDA GEMV fast path
_FPA_INTB_OPTION_INT4 = 0x04
_FPA_INTB_OPTION_INT8 = 0x08

_CACHE_TABLE_SUFFIX = ".matmulnbits_fpa_intb.tsv"


def _parse_m_values(text: str) -> list[int]:
    values = []
    for t in text.split(","):
        token = t.strip()
        if not token:
            continue
        m = int(token)
        if m > 0:
            values.append(m)
    # De-duplicate while keeping ascending order.
    return sorted(set(values))


def _set_tuning_env(output_prefix: str, enable_gemv: bool, m_values: list[int]) -> None:
    """Configures the environment so the CUDA EP writes tuned tactics to disk.

    Must be called before the InferenceSession is created.
    """
    os.environ["ORT_CUDA_GEMM_TACTIC_CACHE_PREFIX"] = output_prefix

    if enable_gemv:
        option = _FPA_INTB_OPTION_ALL
    else:
        # Enable both int4 and int8 GEMM without the CUDA GEMV fast path.
        option = _FPA_INTB_OPTION_INT4 | _FPA_INTB_OPTION_INT8
    os.environ["ORT_FPA_INTB_GEMM"] = str(option)

    if m_values:
        os.environ["ORT_FPA_INTB_PROFILE_M"] = ",".join(str(m) for m in m_values)


def _numpy_dtype_for(ort_type: str):
    mapping = {
        "tensor(float16)": np.float16,
        "tensor(bfloat16)": np.float16,  # numpy has no bf16; only used for dummy inputs
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(ort_type, np.float32)


def _make_dummy_inputs(session, m: int) -> dict:
    """Best-effort dummy inputs. The first symbolic/dynamic dim of each input is set to
    ``m`` (a heuristic to drive the GEMM M dimension); remaining dynamic dims become 1.
    """

    feeds = {}
    for inp in session.get_inputs():
        shape = []
        replaced = False
        for dim in inp.shape:
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            else:
                shape.append(m if not replaced else 1)
                replaced = True
        if not shape:
            shape = [1]
        dtype = _numpy_dtype_for(inp.type)
        if np.issubdtype(dtype, np.floating):
            feeds[inp.name] = np.zeros(shape, dtype=dtype)
        else:
            feeds[inp.name] = np.zeros(shape, dtype=dtype)
    return feeds


def _summarize_cache(cache_path: str) -> None:
    if not os.path.exists(cache_path):
        print(f"WARNING: no cache file was produced at {cache_path}.")
        print("  The model may have no fp16/bf16 MatMulNBits nodes on the fpA_intB path,")
        print("  or the CUDA execution provider was not used.")
        return

    header = {}
    columns = None
    n_key_col = None
    rows = 0
    unique_keys = set()
    with open(cache_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].strip().split("\t")
                if len(parts) >= 2:
                    header[parts[0]] = parts[1]
                continue
            fields = line.split("\t")
            if columns is None:
                columns = fields
                n_key_col = {name: i for i, name in enumerate(columns)}
                continue
            rows += 1
            # Build a key tuple from the problem-key columns for a unique-shape count.
            key_cols = [
                "n_16b",
                "k",
                "activation_dtype",
                "weight_type",
                "bits",
                "block_size",
                "has_zero_points",
                "gemv_enabled",
                "packing_sm",
            ]
            key = tuple(fields[n_key_col[c]] for c in key_cols if c in n_key_col)
            unique_keys.add(key)

    print(f"Cache written: {cache_path}")
    print(f"  device_name      : {header.get('device_name', '?')}")
    print(f"  sm               : {header.get('sm', '?')}")
    print(f"  cuda_runtime     : {header.get('cuda_runtime', '?')}")
    print(f"  ort_version      : {header.get('ort_version', '?')}")
    print(f"  ort_git_commit   : {header.get('ort_git_commit', '?')}")
    print(f"  unique shapes    : {len(unique_keys)}")
    print(f"  tuned (shape, M) : {rows}")


def tune(model: str, output_prefix: str, enable_gemv: bool, m_values: list[int], run_inference: bool) -> str:
    _set_tuning_env(output_prefix, enable_gemv, m_values)

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            f"CUDAExecutionProvider is not available in this onnxruntime build. Available providers: {available}"
        )

    print(f"Creating CUDA session for {model} (this profiles the M buckets)...")
    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(model, sess_options, providers=["CUDAExecutionProvider"])

    if run_inference:
        for m in m_values:
            try:
                feeds = _make_dummy_inputs(session, m)
                session.run(None, feeds)
                print(f"  ran dummy inference for M={m}")
            except Exception as exc:
                print(f"  skipped dummy inference for M={m}: {exc}")

    cache_path = output_prefix + _CACHE_TABLE_SUFFIX
    _summarize_cache(cache_path)
    return cache_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline tactic tuning for MatMulNBits fpA_intB CUDA kernels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to the ONNX model to tune.")
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Cache file prefix. Writes '<output-prefix>.matmulnbits_fpa_intb.tsv'.",
    )
    parser.add_argument(
        "--enable-gemv",
        action="store_true",
        help="Enable the CUDA GEMV fast path when tuning (matches ORT_FPA_INTB_GEMM=1).",
    )
    parser.add_argument(
        "--m-values",
        default="1,2,4,8,16,32,64,128,256,512,1024,2048",
        help="Comma-separated M buckets to profile (sets ORT_FPA_INTB_PROFILE_M).",
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Only trigger construction-time profiling; skip the dummy inference loop.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 2

    m_values = _parse_m_values(args.m_values)
    if not m_values:
        print("ERROR: --m-values did not contain any positive integers.", file=sys.stderr)
        return 2

    output_prefix = os.path.abspath(args.output_prefix)
    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tune(
        model=args.model,
        output_prefix=output_prefix,
        enable_gemv=args.enable_gemv,
        m_values=m_values,
        run_inference=not args.no_inference,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
