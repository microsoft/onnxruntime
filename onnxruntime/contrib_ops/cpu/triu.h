// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/pad.h"

namespace onnxruntime {
namespace contrib {

class Triu final : public OpKernel {
 public:
  explicit Triu(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* ctx) const override;

 private:
   template<typename T>
   struct ComputeImpl;
};

}  // namespace contrib
}  // namespace onnxruntime
