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
class AddGeluFusion final : public BinaryElementwise<ShouldBroadcast> {
 public:
  AddGeluFusion(const OpKernelInfo& info) : BinaryElementwise(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
