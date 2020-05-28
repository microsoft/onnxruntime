// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This module hosts implementations and thin wrappers over other onnx operator implementations
// that will be called from within the Einsum operator implementation

#pragma once

#include "core/providers/cpu/math/einsum_auxiliary_ops.h"
#include "core/providers/cuda/tensor/transpose.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "einsum_auxiliary_ops_diagonal.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {

namespace EinsumOp {

namespace DeviceHelpers {

// These are CUDA EP specific device helper implementations
namespace CudaDeviceHelpers {

Status Transpose(const std::vector<size_t>& permutation, const Tensor& input,
                 Tensor& output, const TensorShape* input_shape_override, void* cublas_handle);

Status DataCopy(const Tensor& input, Tensor& output);

template <typename T>
Status MatMul(const T* input_1_data, const T* input_2_data, T* output_data,
              size_t left_stride, size_t right_stride, size_t output_stride,
              size_t num_batches, size_t M, size_t K, size_t N, concurrency::ThreadPool* tp,
              void* cublas_handle);

template <typename T>
Tensor ReduceSum(const Tensor& input, const std::vector<int64_t>& reduce_axes,
                 bool keep_dims, AllocatorPtr allocator,
                 const TensorShape* input_shape_override,
                 concurrency::ThreadPool* /*tp*/, void* cuda_ep);

std::unique_ptr<Tensor> Diagonal(const Tensor& input, int64_t dim_1, int64_t dim_2, AllocatorPtr allocator);

}  // namespace CudaDeviceHelpers

}  // namespace DeviceHelpers

}  // namespace EinsumOp

}  // namespace onnxruntime
