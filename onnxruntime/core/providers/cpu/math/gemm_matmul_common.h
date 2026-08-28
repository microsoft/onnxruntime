// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {

bool GemmPackBFp32(AllocatorPtr& alloc,
                   const Tensor& tensor_b,
                   bool trans_a,
                   bool trans_b,
                   IAllocatorUniquePtr<void>& packed_b,
                   size_t& packed_b_size,
                   TensorShape& b_shape,
                   const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* mlas_backend_kernel_selector_config);
};  // namespace onnxruntime
