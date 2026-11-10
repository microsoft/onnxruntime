// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void Impl_BitShift(
    cudaStream_t stream,
    int32_t output_rank_or_simple_broadcast,
    const TArray<int64_t>* lhs_padded_strides,
    const T* lhs_data,
    const TArray<int64_t>* rhs_padded_strides,
    const T* rhs_data,
    const TArray<fast_divmod>* fdm_output_strides,
    const fast_divmod& fdm_H,
    const fast_divmod& fdm_C,
    T* output_data,
    size_t count,
    bool shift_left);

}  // namespace cuda
}  // namespace onnxruntime
