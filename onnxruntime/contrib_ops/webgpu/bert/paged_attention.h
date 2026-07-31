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

// WebGPU PagedAttention kernel — v1 skeleton.
//
// Op contract, phased delivery plan, and reuse strategy are documented in
// docs/design/webgpu_paged_attention.md.
//
// This class registers the op with the WebGPU EP so a graph containing
// PagedAttention no longer fails at kernel-matching time. ComputeInternal
// currently returns NOT_IMPLEMENTED; the real implementation is added in
// Phase 1.
class PagedAttention final : public WebGpuKernel {
 public:
  explicit PagedAttention(const OpKernelInfo& info);
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

 private:
  int num_heads_;
  int kv_num_heads_;
  int local_window_size_;
  bool do_rotary_;
  bool rotary_interleaved_;
  float scale_;
  float softcap_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
