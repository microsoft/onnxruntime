// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"
#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/tensor/transpose.h"

namespace onnxruntime {
namespace rocm {

class Transpose final : public RocmKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : RocmKernel(info), TransposeBase(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;

  static Status DoTranspose(const Transpose& transpose_kernel,
                            const std::vector<size_t>& permutations, const Tensor& input, Tensor& output);

  //  `input_shape_override` (if provided) overrides the shape of `input` for compute purposes
  static Status DoTranspose(const hipDeviceProp_t& prop,
                            hipStream_t stream,
                            const rocblas_handle rocblas_handle,
                            const std::vector<size_t>& permutations,
                            const Tensor& input, Tensor& output, const TensorShape* input_shape_override = nullptr);
};

}  // namespace rocm
}  // namespace onnxruntime
