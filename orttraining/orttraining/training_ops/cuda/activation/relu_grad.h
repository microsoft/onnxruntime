// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/activation/activations.h"
#include "orttraining/training_ops/cuda/activation/relu_grad_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class ReluGrad final : public BinaryElementwise<ShouldNotBroadcast> {
 public:
  ReluGrad(const OpKernelInfo& info) : BinaryElementwise(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_NULL()
};
}  // namespace cuda
}  // namespace onnxruntime