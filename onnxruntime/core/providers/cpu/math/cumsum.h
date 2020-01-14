// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/pad.h"

namespace onnxruntime {

template <class T>
class CumSum final : public OpKernel {
 public:
  explicit CumSum(const OpKernelInfo& op_kernel_info);

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  int64_t exclusive_;
  int64_t reverse_;
};

}  // namespace onnxruntime
