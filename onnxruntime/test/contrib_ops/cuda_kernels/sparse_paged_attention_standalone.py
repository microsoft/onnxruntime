#!/usr/bin/env python3
"""Compile the production sparse device kernels without rebuilding the ORT provider."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    source = (root / "onnxruntime/contrib_ops/cuda/bert/sparse_paged_attention_impl.cu").read_text()
    mma = (root / "onnxruntime/contrib_ops/cuda/bert/gated_delta_net_mma.cuh").read_text()
    device = source[source.index("template <typename TCACHE>"):source.index("int SparsePagedAttentionHeadsPerBlock(")]
    helpers = mma[mma.index("__device__ __forceinline__ uint32_t PackHalf2"):mma.index("__device__ __forceinline__ float LoadC")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    header = args.output_dir / "sparse_device.cuh"
    header.write_text(
        "#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <mma.h>\n"
        "#include <cfloat>\n#include <cstdint>\n#include <type_traits>\n"
        "namespace onnxruntime { namespace contrib { namespace cuda {\n"
        "namespace gated_delta_net {\n" + helpers + "\n}\n" + device + "\n}}}\n"
    )
    subprocess.run([
        args.nvcc, "-std=c++17", "-O3", "-arch=sm_90", "-lineinfo",
        "-I" + str(args.output_dir.resolve()),
        str(Path(__file__).with_suffix(".cu")), "-o", str(args.output_dir / "sparse_benchmark"),
    ], check=True)


if __name__ == "__main__":
    main()
