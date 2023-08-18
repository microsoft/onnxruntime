// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
template <class T>
Status Dequantize4Bits(T* output, const uint8_t* quant_data, int K, int N, bool has_zero_point, cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
