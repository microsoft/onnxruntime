// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
//#include "core/providers/cpu/tensor/pad.h"

namespace onnxruntime {
namespace contrib {

template <class T>
class DenseToDenseSetOperation final : public OpKernel {
 public:
  DenseToDenseSetOperation(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    if (!op_kernel_info.GetAttr<int64_t>("default_value", &default_value_).IsOK()) {
      default_value_ = 0;
    }
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 protected:
  int64_t default_value_{0};
};

}  // namespace contrib
}  // namespace onnxruntime
