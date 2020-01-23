// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/activation/activations.h"
#include "training_ops/cuda/activation/activations_grad_impl.h"

//using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

template <typename T>
class GeluGrad final : public BinaryElementwise<ShouldNotBroadcast> {
 public:
  GeluGrad(const OpKernelInfo& info) : BinaryElementwise(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  MAKE_FUNC_CTX_NULL()
};

}  // namespace cuda
}  // namespace onnxruntime
