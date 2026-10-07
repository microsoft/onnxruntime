// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_8bits_batched_impl.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template bool TryMatMul8BitsBatched<nv_bfloat16>(
    nv_bfloat16*, const nv_bfloat16*, const uint8_t*, const nv_bfloat16*, const uint8_t*,
    int, int, int, int, cudaStream_t);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime