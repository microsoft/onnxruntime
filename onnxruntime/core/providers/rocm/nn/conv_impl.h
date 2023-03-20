// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

template <typename T, typename T1, typename T2>
void ConvBiasImpl(
    hipStream_t stream,
    const T1* lhs_data,
    const T2* rhs_data,
    T* output_data,
    size_t bias_size,
    size_t count);

}  // namespace rocm
}  // namespace onnxruntime
