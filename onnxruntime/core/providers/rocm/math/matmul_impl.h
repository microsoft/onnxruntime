// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Modifications: Remove cudaDeviceProp in LaunchFastGeluKernel.
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
Status MatMulImpl(const RocmKernel* op, MatMulComputeHelper& helper,
                  const T* left_x_data, const T* right_x_data, T* output_y_data,
                  const TensorShape& left_shape, const TensorShape& right_shape,
                  bool transa, bool transb, bool trans_batch_a, bool trans_batch_b,
                  const float t_alpha, const float t_zero);

}  // namespace rocm
}  // namespace onnxruntime
