// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

bool GemmPackBFp32(AllocatorPtr& alloc,
                   const Tensor& tensor_b,
                   bool trans_b,
                   BufferUniquePtr& packed_b,
                   TensorShape& b_shape);

};  // namespace onnxruntime
