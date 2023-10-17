#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class StringConcat final : public OpKernel {
 public:
  StringConcat(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
