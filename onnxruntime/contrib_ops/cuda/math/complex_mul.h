// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/math/binary_elementwise_ops.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace ::onnxruntime::cuda;

template <typename T>
class ComplexMul : public BinaryElementwise<ShouldBroadcast> {
 public:
  ComplexMul(const OpKernelInfo info) : BinaryElementwise{info} {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class ComplexMulConj : public BinaryElementwise<ShouldBroadcast> {
 public:
  ComplexMulConj(const OpKernelInfo info) : BinaryElementwise{info} {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
