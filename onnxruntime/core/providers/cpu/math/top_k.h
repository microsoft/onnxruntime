// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/util/math_cpuonly.h"
#include "gsl/gsl_util"

namespace onnxruntime {
template <typename T>
class TopK final : public OpKernel {
 public:
  TopK(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    int64_t k_temp;
    ONNXRUNTIME_ENFORCE(op_kernel_info.GetAttr<int64_t>("k", &k_temp).IsOK());
    ONNXRUNTIME_ENFORCE(k_temp > 0);
    k_ = gsl::narrow_cast<unsigned>(k_temp);

    int64_t axis_temp;
    ONNXRUNTIME_ENFORCE(op_kernel_info.GetAttr<int64_t>("axis", &axis_temp).IsOK());
    axis_ = gsl::narrow_cast<int>(axis_temp);
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  int axis_;
  unsigned k_;
};
}  // namespace onnxruntime
