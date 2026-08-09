// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace cuda {
template <typename T, bool Layout>
void MaxPoolWithIndex(
    cudaStream_t stream,
    const TensorShape& input_shape,
    const TensorShape& output_shape,
    const gsl::span<const int64_t>& kernel_shape,
    const gsl::span<const int64_t>& stride_shape,
    const gsl::span<const int64_t>& pads,
    const gsl::span<const int64_t>& dilations,
    int64_t storage_order,
    const T* p_input,
    T* p_output,
    int64_t* p_indices);
}  // namespace cuda
}  // namespace onnxruntime
