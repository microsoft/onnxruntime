// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace contrib {
namespace hip {

using namespace onnxruntime::hip;

template <typename T>
class FastGelu final : public HipKernel {
 public:
  FastGelu(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace hip
}  // namespace contrib
}  // namespace onnxruntime
