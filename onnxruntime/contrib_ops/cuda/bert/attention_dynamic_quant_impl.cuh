// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class Tin>
Status CudaQuantizeLinearSimple(const Tin* input, int8_t* output, float scale, int num_of_element);

template <class Tin>
Status CudaDequantizeWithBias(const int32_t* quantize, const Tin* bias, Tin* output, Tin scale, int m, int n);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
