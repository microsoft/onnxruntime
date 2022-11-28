// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "contrib_ops/rocm/bert/fast_gelu_impl.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

template <typename T>
class FastGelu final : public RocmKernel {
 public:
  FastGelu(const OpKernelInfo& op_kernel_info) : RocmKernel(op_kernel_info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
