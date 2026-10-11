// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_4bits_batched_impl.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template bool TryMatMulBatched4Bits<half>(
    half*, const half*, const uint8_t*, const half*, const uint8_t*,
    int, int, int, int, size_t, cudaStream_t);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime