// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/math/binary_elementwise_ops.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fast_divmod.h"
#include "core/providers/cpu/tensor/utils.h"

using namespace onnxruntime::rocm;

namespace onnxruntime {
namespace contrib {
namespace rocm {

// AddGelu fuse Add + Gelu
template <typename T>
class BiasGelu final : public BinaryElementwise<ShouldBroadcast> {
 public:
  BiasGelu(const OpKernelInfo& info) : BinaryElementwise(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
