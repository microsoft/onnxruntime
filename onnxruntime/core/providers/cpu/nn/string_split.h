#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class StringSplit final : public OpKernel {
 public:
  explicit StringSplit(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  std::string delimiter_;
  int64_t maxsplit_;
};

}  // namespace onnxruntime
