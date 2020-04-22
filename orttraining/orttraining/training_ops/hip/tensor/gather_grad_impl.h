// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/hip/hip_common.h"
#include "core/providers/hip/shared_inc/hip_utils.h"

namespace onnxruntime {
namespace hip {

template <typename T, typename Tin>
void GatherGradImpl(
    const HipKernel& hip_kernel,
    const T* grad_data,
    const Tin* indices_data,
    const int64_t num_indices,
    const int64_t num_weights,
    const int64_t stride,
    T* output_data,
    const int64_t num_inputs,
    const int64_t param_itrs);

}  // namespace hip
}  // namespace onnxruntime
