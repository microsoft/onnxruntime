# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import statistics
import time
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxruntime as ort


@dataclass(frozen=True)
class EinsumCase:
    name: str
    equation: str
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]


CASES = (
    EinsumCase("copy", "bhsd->bhsd", ((1, 16, 128, 64),), (1, 16, 128, 64)),
    EinsumCase("transpose", "bhsd->bshd", ((1, 16, 128, 64),), (1, 128, 16, 64)),
    EinsumCase("reduce", "bsk->bs", ((4, 128, 1024),), (4, 128)),
    EinsumCase("diagonal", "bii->bi", ((16, 512, 512),), (16, 512)),
    EinsumCase("trace", "bii->b", ((16, 512, 512),), (16,)),
    EinsumCase("multiply", "bsh,bsh->bsh", ((4, 128, 768), (4, 128, 768)), (4, 128, 768)),
    EinsumCase("outer", "i,j->ij", ((1024,), (1024,)), (1024, 1024)),
    EinsumCase("dot", "i,i->", ((4 * 1024 * 1024,), (4 * 1024 * 1024,)), ()),
    EinsumCase("gemm", "mk,kn->mn", ((2048, 1024), (1024, 1024)), (2048, 1024)),
    EinsumCase(
        "attention_qk",
        "bhmd,bhnd->bhmn",
        ((1, 16, 128, 64), (1, 16, 128, 64)),
        (1, 16, 128, 128),
    ),
    EinsumCase(
        "fallback_three_input",
        "bik,bkj,bjl->bil",
        ((4, 128, 256), (4, 256, 128), (4, 128, 64)),
        (4, 128, 64),
    ),
)


def create_model(case: EinsumCase, tensor_type: int) -> bytes:
    input_names = [f"X{i}" for i in range(len(case.input_shapes))]
    node = helper.make_node("Einsum", input_names, ["Y"], equation=case.equation)
    graph = helper.make_graph(
        [node],
        f"einsum_{case.name}",
        [
            helper.make_tensor_value_info(name, tensor_type, shape)
            for name, shape in zip(input_names, case.input_shapes, strict=False)
        ],
        [helper.make_tensor_value_info("Y", tensor_type, case.output_shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = onnx.IR_VERSION
    return model.SerializeToString()


def benchmark_case(case: EinsumCase, dtype, tensor_type: int, args) -> tuple[float, float]:
    provider_options = {"enable_cuda_graph": "1"} if args.cuda_graph else {}
    session = ort.InferenceSession(
        create_model(case, tensor_type),
        providers=[("CUDAExecutionProvider", provider_options)],
    )
    rng = np.random.default_rng(0)
    inputs = [
        ort.OrtValue.ortvalue_from_numpy(rng.standard_normal(shape).astype(dtype), "cuda", 0)
        for shape in case.input_shapes
    ]
    output = ort.OrtValue.ortvalue_from_shape_and_type(case.output_shape, dtype, "cuda", 0)
    binding = session.io_binding()
    for index, value in enumerate(inputs):
        binding.bind_ortvalue_input(f"X{index}", value)
    binding.bind_ortvalue_output("Y", output)

    for _ in range(args.warmup):
        session.run_with_iobinding(binding)
    binding.synchronize_outputs()

    samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        samples.append((time.perf_counter() - start) * 1000)

    return statistics.median(samples), float(np.percentile(samples, 90))


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA Einsum fast paths and fallback")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--cuda_graph", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--case", action="append", choices=[case.name for case in CASES])
    args = parser.parse_args()

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is required")

    dtype = np.float16 if args.precision == "fp16" else np.float32
    tensor_type = TensorProto.FLOAT16 if args.precision == "fp16" else TensorProto.FLOAT
    selected_cases = [case for case in CASES if not args.case or case.name in args.case]
    print(f"provider=CUDAExecutionProvider precision={args.precision} cuda_graph={args.cuda_graph}")
    for case in selected_cases:
        median_ms, p90_ms = benchmark_case(case, dtype, tensor_type, args)
        print(f"{case.name:24s} equation={case.equation:20s} median={median_ms:9.4f} ms p90={p90_ms:9.4f} ms")


if __name__ == "__main__":
    main()
