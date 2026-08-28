// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace cuda {

// Custom average-pooling CUDA kernel that honors per-side (asymmetric) padding.
//
// cuDNN's pooling descriptor stores a single symmetric pad value per axis, so it cannot
// represent ONNX asymmetric padding (pad_begin != pad_end), which arises from explicit
// asymmetric `pads` or `auto_pad = SAME_UPPER/SAME_LOWER` resolving to asymmetric pads. This
// kernel is the CUDA fallback for that case and mirrors the CPU reference functor
// (AveragePool{1,2,3}DTask) exactly:
//   start = out_idx * stride - pad_begin
//   end   = min(start + kernel * dilation, in_size + pad_end)
//   sum over cells [start, end) with dilation step that are in [0, in_size)
//   count_include_pad == 1: divisor = product of (1 + (end - start - 1) / dilation)
//   count_include_pad == 0: divisor = number of summed in-bounds cells
//
// `pads` is the full ONNX layout [x1_begin,...,xN_begin, x1_end,...,xN_end] (2 * rank).
template <typename T, bool Layout>
void AveragePoolWithPad(
    cudaStream_t stream,
    const TensorShape& input_shape,
    const TensorShape& output_shape,
    const gsl::span<const int64_t>& kernel_shape,
    const gsl::span<const int64_t>& stride_shape,
    const gsl::span<const int64_t>& pads,
    const gsl::span<const int64_t>& dilations,
    bool count_include_pad,
    const T* p_input,
    T* p_output);

}  // namespace cuda
}  // namespace onnxruntime
