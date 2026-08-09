// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/pad.h"

namespace onnxruntime {
namespace contrib {

template <class T>
class Unique final : public OpKernel {
 public:
  explicit Unique(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {}

  Status Compute(OpKernelContext* p_op_kernel_context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime