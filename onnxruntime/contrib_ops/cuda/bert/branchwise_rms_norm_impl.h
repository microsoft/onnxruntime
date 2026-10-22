// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

#include "core/common/status.h"

namespace onnxruntime::contrib::cuda {

template <typename T>
Status LaunchMixedScaleBranchwiseRMSNorm(cudaStream_t stream, const T* x, const void* scale,
                                         int scale_type, T* y, int64_t groups, int branches,
                                         int hidden, bool shared_scale, float epsilon);

template <typename T>
Status LaunchStandardBranchwiseRMSNorm(cudaStream_t stream, const T* x, const void* scale,
                                       int scale_type, T* y, int groups, int branches,
                                       int hidden, bool shared_scale, float epsilon,
                                       float* widened, float* squares, float* sums, void* reduction_buffer,
                                       size_t reduction_buffer_size, int input_row_stride,
                                       bool deterministic_compute, bool reciprocal_multiply = false);

}  // namespace onnxruntime::contrib::cuda
