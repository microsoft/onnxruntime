// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/hip/hip_utils.h"
#include "orttraining/training_ops/hip/tensor/thrustallocator.h"

namespace onnxruntime {
namespace hip {

template <typename T, typename Tin>
void GatherGradImpl(
    const T* grad_data,
    const Tin* indices_data,
    const int64_t num_indices,
    const int64_t num_weights,
    const int64_t stride,
    Tin* origin_indices,
    T* output_data,
    const int64_t num_inputs,
    const int64_t param_itrs,
    ThrustAllocator& allocator);

}  // namespace hip
}  // namespace onnxruntime
