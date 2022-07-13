// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using onnxruntime::rocm::RocmKernel;

template <typename T>
class GemmFastGelu final : public RocmKernel {
 public:
  GemmFastGelu(const OpKernelInfo& op_kernel_info) : RocmKernel(op_kernel_info) {}
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
