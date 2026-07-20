// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace webgpu {

class RotaryEmbedding final : public WebGpuKernel {
 public:
  RotaryEmbedding(const OpKernelInfo& info);
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int num_heads_;
  int rotary_embedding_dim_;
  bool interleaved_;
};

}  // namespace webgpu
}  // namespace onnxruntime
