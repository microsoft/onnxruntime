// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_TORCH

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class TorchEmbeddingGrad final : public OpKernel {
 public:
  TorchEmbeddingGrad(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* p_ctx) const override;
};

}  // namespace contrib
}  // namespace onnxruntime

#endif
