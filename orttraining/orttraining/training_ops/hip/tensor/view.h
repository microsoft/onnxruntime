// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/hip/hip_common.h"

namespace onnxruntime {
namespace hip {

class View final : public HipKernel {
 public:
  View(const OpKernelInfo& info) : HipKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace hip
}  // namespace onnxruntime
