// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class AdasumAllReduce final : public OpKernel {
 public:
  explicit AdasumAllReduce(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
