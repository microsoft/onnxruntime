// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/hip/hip_common.h"

using namespace onnxruntime::hip;

namespace onnxruntime {
namespace contrib {
namespace hip {

template <typename T, typename U>
class LayerNorm final : public HipKernel {
 public:
  LayerNorm(const OpKernelInfo& op_kernel_info);

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  int64_t axis_;
  double epsilon_;
};

}  // namespace hip
}  // namespace contrib
}  // namespace onnxruntime
