# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Offline tactic tuning tool for the MatMulNBits fpA_intB CUDA path.

This tool tunes the weight-only GEMM tactics for a model's ``MatMulNBits`` nodes and
writes the results to a persistent tactic cache file that ONNX Runtime can reuse on
later runs (see docs/contrib_ops/cuda/gemm_profiler_cache.md).

How it works:
  * It creates a CUDA execution-provider session for the model with the fpA_intB path and the
    cache prefix enabled through session config entries. Kernel construction profiles the
    configured M buckets and stages their successful tactics in memory.
  * It then (best-effort) runs dummy inferences at each requested M value so that any
    additional buckets are profiled lazily. All staged tactics are written to
    ``<output-prefix>.matmulnbits_fpa_intb.tsv`` when the session is released.

The generated cache is hardware/build specific: it is only reused on the same GPU
model + SM + CUDA runtime + ORT version.

Example:
    python -m onnxruntime.tools.fpa_intb_tune \
        --model model.onnx \
        --output-prefix /path/to/cache/mymodel \
        --m-values 1,8,16,32,64,128,256,512,1024,2048
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import sys
from urllib.parse import unquote

import numpy as np

import onnxruntime as ort

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


def _make_session_options(output_prefix: str, m_values: list[int]) -> ort.SessionOptions:
    """Enables the fpA_intB path, its M-bucket sweep, and the persistent cache for one session."""
    sess_options = ort.SessionOptions()
    sess_options.add_session_config_entry("ep.cuda.gemm_tactic_cache_prefix", output_prefix)
    sess_options.add_session_config_entry("ep.cuda.fpa_intb_gemm", "1")
    if m_values:
        sess_options.add_session_config_entry("ep.cuda.fpa_intb_profile_m", ",".join(str(m) for m in m_values))
    return sess_options


def _numpy_dtype_for(ort_type: str):
    mapping = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    if ort_type not in mapping:
        raise ValueError(f"Unsupported dummy input type: {ort_type}")
    return mapping[ort_type]


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
        if inp.type == "tensor(bfloat16)":
            # BF16 zero has the same bits as uint16 zero; NumPy extension dtypes are not accepted by run().
            feeds[inp.name] = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(np.zeros(shape, dtype=np.uint16), 16)
        else:
            feeds[inp.name] = ort.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=_numpy_dtype_for(inp.type)))
    return feeds


def _current_signature(device_id: int) -> dict[str, str]:
    # Use the CUDA driver for device identity and the wheel metadata for the toolkit it was built with.
    from onnxruntime.capi.build_and_package_info import cuda_version  # noqa: PLC0415

    driver = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")

    def check(code):
        if code != 0:
            raise RuntimeError(f"Cannot query the CUDA cache signature (CUDA driver error {code})")

    check(driver.cuInit(0))
    device = ctypes.c_int()
    check(driver.cuDeviceGet(ctypes.byref(device), device_id))
    name = ctypes.create_string_buffer(256)
    check(driver.cuDeviceGetName(name, len(name), device))
    major, minor = ctypes.c_int(), ctypes.c_int()
    check(driver.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device))
    toolkit = cuda_version.split(".")
    return {
        "device_name": name.value.decode("utf-8"),
        "sm": str(major.value * 10 + minor.value),
        "cuda_runtime": str(int(toolkit[0]) * 1000 + int(toolkit[1]) * 10),
        "ort_version": ort.__version__,
    }


def _summarize_cache(cache_path: str, signature: dict[str, str]) -> None:
    if not os.path.exists(cache_path):
        raise RuntimeError(
            f"No tactic cache was produced at {cache_path}; check that the model uses fpA_intB MatMulNBits."
        )

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
                    header[parts[0]] = unquote(parts[1])
                continue
            fields = line.split("\t")
            if columns is None:
                columns = fields
                n_key_col = {name: i for i, name in enumerate(columns)}
                continue
            if len(fields) != len(columns) or fields[n_key_col["valid_config"]] != "1":
                raise RuntimeError(f"Invalid or unsuccessful tactic in {cache_path}")
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
                "has_bias",
                "packing_sm",
            ]
            key = tuple(fields[n_key_col[c]] for c in key_cols if c in n_key_col)
            unique_keys.add(key)

    expected = {
        **signature,
        "ort_cuda_gemm_tactic_cache": "v1",
        "table": "matmulnbits_fpa_intb",
        "tactic_selection_version": "2",
    }
    for key, value in expected.items():
        if header.get(key) != value:
            raise RuntimeError(f"Inapplicable tactic cache: {key}={header.get(key)!r}, expected {value!r}")
    if rows == 0:
        raise RuntimeError(f"No successful tactics in {cache_path}")
    print(f"Cache ready: {cache_path}")
    print(f"  device_name      : {header.get('device_name', '?')}")
    print(f"  sm               : {header.get('sm', '?')}")
    print(f"  cuda_runtime     : {header.get('cuda_runtime', '?')}")
    print(f"  ort_version      : {header.get('ort_version', '?')}")
    print(f"  ort_git_commit   : {header.get('ort_git_commit', '?')}")
    print(f"  unique shapes    : {len(unique_keys)}")
    print(f"  tuned (shape, M) : {rows}")


def tune(model: str, output_prefix: str, m_values: list[int], run_inference: bool) -> str:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            f"CUDAExecutionProvider is not available in this onnxruntime build. Available providers: {available}"
        )

    print(f"Creating CUDA session for {model} (this profiles the M buckets)...")
    sess_options = _make_session_options(output_prefix, m_values)
    session = ort.InferenceSession(model, sess_options, providers=["CUDAExecutionProvider"], enable_fallback=False)
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("Tuning session did not activate CUDAExecutionProvider")
    session.disable_fallback()
    device_id = int(session.get_provider_options().get("CUDAExecutionProvider", {}).get("device_id", "0"))
    signature = _current_signature(device_id)

    if run_inference:
        for m in m_values:
            try:
                feeds = _make_dummy_inputs(session, m)
                session.run_with_ort_values(None, feeds)
                print(f"  ran dummy inference for M={m}")
            except Exception as exc:
                print(f"  skipped dummy inference for M={m}: {exc}")

    # Lazily profiled buckets reach disk only when the CUDA execution provider is torn down.
    del session
    gc.collect()

    cache_path = output_prefix + _CACHE_TABLE_SUFFIX
    _summarize_cache(cache_path, signature)
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
        "--m-values",
        default="1,2,4,8,16,32,64,128,256,512,1024,2048",
        help="Comma-separated M buckets to profile (sets ep.cuda.fpa_intb_profile_m).",
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
        m_values=m_values,
        run_inference=not args.no_inference,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
