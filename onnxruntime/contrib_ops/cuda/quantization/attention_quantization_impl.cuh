// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
template <class Tin>
Status CudaDequantizeWithBias(cudaStream_t stream, const int32_t* quantize, const Tin* bias, Tin* output, Tin scale, int m, int n);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
