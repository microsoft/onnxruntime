// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/providers/cpu/tensor/utils.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// AddGelu fuse Add + Gelu
template <typename T>
class BiasGelu final : public BinaryElementwise<ShouldBroadcast> {
 public:
  BiasGelu(const OpKernelInfo& info) : BinaryElementwise(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

//Onnx BitShift does not support int32_t and int64_t
template <typename T>
class BitShift final : public BinaryElementwise<ShouldBroadcast> {
 public:
  BitShift(const OpKernelInfo& info) : BinaryElementwise(info) {
    std::string direction;
    ORT_THROW_IF_ERROR(info.GetAttr<std::string>("direction", &direction));
    right_shift_ = direction == "RIGHT";
  }
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool right_shift_{false};
};

template <typename T>
class BitwiseAnd final : public BinaryElementwise<ShouldBroadcast> {
 public:
  BitwiseAnd(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
