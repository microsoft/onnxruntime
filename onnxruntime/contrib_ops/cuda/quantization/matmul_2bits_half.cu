// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_2bits_m1_impl.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template bool TryMatMul2BitsM1<half>(
    half*, const half*, const uint8_t*, const half*, const uint8_t*, int, int, int, size_t, cudaStream_t);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
