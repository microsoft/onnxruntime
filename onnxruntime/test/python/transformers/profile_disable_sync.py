#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Profiling helper to verify that the CUDA EP does NOT perform any host-side
stream/device synchronization during Run() when synchronization is disabled
via the run config entry `disable_synchronize_execution_providers=1`
(added in https://github.com/microsoft/onnxruntime/pull/28686).

The script builds a tiny multi-op CUDA model, binds inputs and outputs to the
GPU (so no implicit device-to-host copies happen during a run), and wraps each
Run() call in an NVTX range. When profiled with nsys, the CUDA runtime API
calls that land inside that NVTX range can then be inspected with
parse_nsys.py --cuda-api --sync-apis-only to confirm there is no
cudaStreamSynchronize / cudaDeviceSynchronize / etc. inside the run.

This is NOT a CI unit test; it requires a CUDA build, a GPU, and (for the full
verification) the `nsys` profiler. See run_disable_sync_check.sh for the
end-to-end driver.

Standalone usage (does not need nsys, just exercises the run):
  python profile_disable_sync.py --sync off --warmup 5 --repeat 50
  python profile_disable_sync.py --sync on  --warmup 5 --repeat 50

Under nsys (so CUDA API + NVTX ranges are captured):
  nsys profile -t cuda,nvtx -o sync_off --export=sqlite \
      python profile_disable_sync.py --sync off --nvtx-range ort_run \
      --warmup 5 --repeat 50
  python parse_nsys.py sync_off.sqlite --cuda-api --sync-apis-only \
      --nvtx-range ort_run --skip-first-ranges 5
"""

import argparse
import os
import sys

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save

import onnxruntime as ort

try:
    import nvtx  # type: ignore
except ImportError:  # pragma: no cover - nvtx is optional for a dry run
    nvtx = None


def build_model(path: str, hidden: int, layers: int, op_type: str = "elementwise") -> None:
    """Build a small CUDA model that launches several kernels per run.

    op_type:
      "elementwise" - a chain of Mul + Add + Relu ops. These map to ORT CUDA
                      elementwise kernels and do NOT call cuBLAS/cuDNN, whose
                      internal host synchronization would otherwise mask the
                      EP-level synchronization that this test isolates.
      "matmul"      - a chain of MatMul + Relu ops (uses cuBLAS).
    """
    rng = np.random.default_rng(0)

    nodes = []
    initializers = []
    current = "input"

    if op_type == "matmul":
        for i in range(layers):
            w_name = f"W{i}"
            mm_out = f"mm{i}"
            relu_out = f"relu{i}"
            weight = rng.standard_normal((hidden, hidden)).astype(np.float32) * (1.0 / hidden**0.5)
            initializers.append(numpy_helper.from_array(weight, name=w_name))
            nodes.append(helper.make_node("MatMul", [current, w_name], [mm_out], name=f"MatMul_{i}"))
            nodes.append(helper.make_node("Relu", [mm_out], [relu_out], name=f"Relu_{i}"))
            current = relu_out
    else:
        # Elementwise chain: (x * scale) + bias -> Relu, repeated.
        scale = numpy_helper.from_array(np.full((hidden,), 1.001, dtype=np.float32), name="scale")
        bias = numpy_helper.from_array(np.full((hidden,), 0.01, dtype=np.float32), name="bias")
        initializers.extend([scale, bias])
        for i in range(layers):
            mul_out = f"mul{i}"
            add_out = f"add{i}"
            relu_out = f"relu{i}"
            nodes.append(helper.make_node("Mul", [current, "scale"], [mul_out], name=f"Mul_{i}"))
            nodes.append(helper.make_node("Add", [mul_out, "bias"], [add_out], name=f"Add_{i}"))
            nodes.append(helper.make_node("Relu", [add_out], [relu_out], name=f"Relu_{i}"))
            current = relu_out

    # Rename last output to "output".
    nodes[-1].output[0] = "output"

    graph = helper.make_graph(
        nodes,
        "disable_sync_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", hidden])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", hidden])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 10
    save(model, path)


def run(args: argparse.Namespace) -> int:
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("ERROR: CUDAExecutionProvider is not available in this onnxruntime build.", file=sys.stderr)
        return 2

    model_path = args.model
    if not os.path.exists(model_path):
        build_model(model_path, args.hidden, args.layers, args.op_type)

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=[("CUDAExecutionProvider", {"device_id": args.device_id})],
    )

    device = "cuda"
    batch = args.batch
    hidden = args.hidden

    input_np = np.random.default_rng(1).standard_normal((batch, hidden)).astype(np.float32)
    # Keep input and output on the GPU so a Run() never needs a device-to-host copy.
    input_ortvalue = ort.OrtValue.ortvalue_from_numpy(input_np, device, args.device_id)
    output_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type([batch, hidden], np.float32, device, args.device_id)

    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input("input", input_ortvalue)
    io_binding.bind_ortvalue_output("output", output_ortvalue)

    sync_disabled = args.sync == "off"
    run_options = ort.RunOptions()
    if sync_disabled:
        run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")

    print(
        f"Model: {model_path} | layers={args.layers} hidden={hidden} batch={batch} | "
        f"sync={'DISABLED' if sync_disabled else 'enabled'} | warmup={args.warmup} repeat={args.repeat}"
    )

    def one_run():
        session.run_with_iobinding(io_binding, run_options)

    # Warmup. These iterations are wrapped in the NVTX range like the measured
    # ones, but parse_nsys.py drops them with --skip-first-ranges, so they do not
    # affect the in-range CUDA API check.
    for _ in range(args.warmup):
        with _nvtx_range(args.nvtx_range):
            one_run()

    for _ in range(args.repeat):
        with _nvtx_range(args.nvtx_range):
            one_run()

    # Final explicit synchronization so the process exits cleanly and the bound
    # output is valid. This synchronizes the output buffers without launching an
    # extra inference run, so it does not pollute the in-range CUDA API check.
    io_binding.synchronize_outputs()

    print("Done.")
    return 0


class _NullRange:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _nvtx_range(name: str):
    if nvtx is None:
        return _NullRange()
    return nvtx.annotate(message=name, color="green")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sync", choices=["on", "off"], default="off", help="Enable or disable EP synchronization")
    parser.add_argument("--model", default="disable_sync_test.onnx", help="Path to generated/used ONNX model")
    parser.add_argument("--nvtx-range", default="ort_run", help="NVTX range name wrapping each Run()")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=50, help="Measured iterations")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=8, help="Number of repeated op blocks")
    parser.add_argument(
        "--op-type",
        choices=["elementwise", "matmul"],
        default="elementwise",
        help="elementwise (no cuBLAS/cuDNN sync; default) or matmul (uses cuBLAS)",
    )
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
