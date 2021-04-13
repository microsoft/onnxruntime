// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(USE_TORCH)

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

class TorchEmbedding final : public OpKernel {
 public:
  TorchEmbedding(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* p_ctx) const override;
};

}  // namespace contrib
}  // namespace onnxruntime

#endif
