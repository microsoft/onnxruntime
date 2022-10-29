// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/nn/layer_norm_impl.h"

namespace onnxruntime {
namespace contrib {

template <bool simplified>
class LayerNorm final : public LayerNormImpl {
 public:
  LayerNorm(const OpKernelInfo& op_kernel_info)
      : LayerNormImpl(op_kernel_info, simplified) {}
};

}  // namespace contrib
}  // namespace onnxruntime
