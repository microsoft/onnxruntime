// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "quantize_linear.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

template <class T, class U>
Status CudaQuantizeLinear(const U* input, T* output, const U* scale, const T* zero_point, size_t num_of_element);

template <class T, class U>
Status CudaDequantizeLinear(const T* input, U* output, const U* scale, const T* zero_point, size_t num_of_element);

}  // namespace cuda
}  // namespace onnxruntime
