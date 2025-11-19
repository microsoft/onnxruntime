// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/layer_norm_impl.h"

namespace onnxruntime {

class RMSNorm final : public LayerNormImpl {
 public:
  RMSNorm(const OpKernelInfo& op_kernel_info)
      : LayerNormImpl(op_kernel_info, /* simplified */ true) {}
};

}  // namespace onnxruntime
