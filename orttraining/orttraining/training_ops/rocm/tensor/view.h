// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

class View final : public RocmKernel {
 public:
  View(const OpKernelInfo& info) : RocmKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
