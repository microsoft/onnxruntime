// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

class Transpose final : public HipKernel, public TransposeBase {
 public:
  Transpose(const OpKernelInfo& info) : HipKernel(info), TransposeBase(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

  static Status DoTranspose(const Transpose& transpose_kernel,
                            const std::vector<size_t>& permutations, const Tensor& input, Tensor& output);
};

}  // namespace hip
}  // namespace onnxruntime
