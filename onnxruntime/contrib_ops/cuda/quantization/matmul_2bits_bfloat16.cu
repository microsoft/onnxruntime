// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_2bits_m1_impl.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template bool TryMatMul2BitsM1<nv_bfloat16>(
    nv_bfloat16*, const nv_bfloat16*, const uint8_t*, const nv_bfloat16*, const uint8_t*, int, int, int, size_t, cudaStream_t);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
