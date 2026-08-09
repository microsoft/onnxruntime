// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "complex_mul.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace ::onnxruntime::cuda;

template <typename T>
void ComplexMul_Impl(
    cudaStream_t stream,
    int32_t output_rank_or_simple_broadcast,
    const TArray<int64_t>* lhs_padded_strides,
    const T* lhs_data,
    const TArray<int64_t>* rhs_padded_strides,
    const T* rhs_data,
    const TArray<onnxruntime::cuda::fast_divmod>* fdm_output_strides,
    const onnxruntime::cuda::fast_divmod& fdm_H,
    const onnxruntime::cuda::fast_divmod& fdm_C,
    T* output_data,
    int64_t count,
    int64_t lhs_size,
    int64_t rhs_size,
    bool is_conj);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
