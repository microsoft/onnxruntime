// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

//#include "core/common/exceptions.h"
#include "core/framework/op_kernel.h"
//#include "core/framework/tensor.h"
//#include "core/util/math_cpuonly.h"
//#include "gsl/gsl_util"

namespace onnxruntime {
template <int OpSet, typename T>
class TopK final : public OpKernel {
 public:
  TopK(const OpKernelInfo& op_kernel_info);

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 private:
  int axis_;
  unsigned k_;
};
}  // namespace onnxruntime
