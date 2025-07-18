// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/cuda_kernel.h"
#include "gather_block_quantized.h"

namespace onnxruntime {
namespace cuda {

template <typename T2, typename Tind>
Status GatherBlockQuantizedImpl(
    cudaStream_t stream,
    const CudaKernel& kernel,
    const Tensor* data,
    const Tensor* indices,
    const Tensor* scales,
    const Tensor* zero_points,
    Tensor& output,
    const int64_t gather_axis,
    const int64_t quantize_axis,
    const int64_t block_size,
    const int64_t bits);

}  // namespace cuda
}  // namespace onnxruntime
