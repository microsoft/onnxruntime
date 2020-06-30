// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class MaskIndex final : public OpKernel {
 public:
  MaskIndex(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
