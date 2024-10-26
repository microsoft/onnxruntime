// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"
#include "contrib_ops/webgpu/bert/webgpu_attention_common.h"
#include "contrib_ops/webgpu/bert/multihead_attention.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class GroupQueryAttention final : public WebGPUKernel, public GQAAttentionBase {
 public:
  GroupQueryAttention(const OpKernelInfo& info) : WebGPUKernel(info), GQAAttentionBase(info, true) {
  }
  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;

};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
