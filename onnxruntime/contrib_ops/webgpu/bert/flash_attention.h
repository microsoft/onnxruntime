// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/webgpu/bert/attention_common.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

class CopyKVCacheProgram final : public Program<CopyKVCacheProgram> {
 public:
  CopyKVCacheProgram(const std::string& kernel_name, bool has_past, bool kv_BNSH, bool past_present_share_buffer)
      : Program{kernel_name}, has_past_(has_past), kv_BNSH_(kv_BNSH), past_present_share_buffer_(past_present_share_buffer) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"valid_present_size", ProgramUniformVariableDataType::Uint32},
                                          {"past_sequence_length", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_past_;
  bool kv_BNSH_;
  bool past_present_share_buffer_;
};

class FlashAttentionProgram final : public Program<FlashAttentionProgram> {
 public:
  FlashAttentionProgram(const std::string& kernel_name,
                        bool has_attention_bias,
                        int qkv_head_size,
                        int qkv_num_heads)
      : Program{kernel_name},
        has_attention_bias_(has_attention_bias),
        qkv_head_size_(qkv_head_size),
        qkv_num_heads_(qkv_num_heads) {
  }

  Status GenerateShaderCode(ShaderHelper& sh) const override;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"new_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"present_sequence_length", ProgramUniformVariableDataType::Uint32},
                                          {"is_gqa", ProgramUniformVariableDataType::Uint32},
                                          {"n_reps", ProgramUniformVariableDataType::Uint32},
                                          {"alpha", ProgramUniformVariableDataType::Float32});

 private:
  bool has_attention_bias_;
  int qkv_head_size_;
  int qkv_num_heads_;
};

Status ApplyFlashAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                           Tensor* output, const Tensor* past_key, Tensor* present_key, const Tensor* past_value, Tensor* present_value,
                           const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context);

bool CanApplyFlashAttention(const Tensor* bias, const Tensor* present_key, const Tensor* present_value,
                            const WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context);
}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
