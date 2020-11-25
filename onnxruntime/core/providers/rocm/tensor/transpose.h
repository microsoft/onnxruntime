// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

class Transpose final : public RocmKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : RocmKernel(info), TransposeBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

  static Status DoTranspose(const Transpose& transpose_kernel,
                            const std::vector<size_t>& permutations, const Tensor& input, Tensor& output);
};

}  // namespace rocm
}  // namespace onnxruntime
