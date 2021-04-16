// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

bool GemmPackBFp32_sizes(const Tensor& tensor_b,
                         bool trans_b,
                         TensorShape& b_shape,
                         size_t& out_packed_b_size,
                         size_t& out_K,
                         size_t& out_N);

bool GemmPackBFp32_run(AllocatorPtr alloc,
                       const Tensor& tensor_b,
                       bool trans_b,
                       BufferUniquePtr& packed_b,
                       size_t packed_b_size,
                       size_t K,
                       size_t N);

bool GemmPackBFp32(const OpKernelInfo& info,
                   const Tensor& tensor_b,
                   bool trans_b,
                   BufferUniquePtr& packed_b,
                   TensorShape& b_shape);

};  // namespace onnxruntime
