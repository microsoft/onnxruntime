// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class MultiHeadAttention final : public WebGpuKernel {
 public:
  MultiHeadAttention(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 protected:
  int num_heads_;
  float mask_filter_value_;
  float scale_;
  bool is_unidirectional_{false};
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
