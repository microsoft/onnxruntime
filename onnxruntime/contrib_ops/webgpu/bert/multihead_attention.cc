// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/webgpu/bert/multihead_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

#include "core/providers/webgpu/webgpu_supported_types.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::multihead_attention_helper;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    MultiHeadAttention,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    MultiHeadAttention);

Status TransferBSDToBNSHProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("qkv_input", ShaderUsage::UseUniform);
  const auto& qkv_output = shader.AddOutput("qkv_output", ShaderUsage::UseUniform | ShaderUsage::UseOffsetToIndices);

  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.data_size")
                            << "let output_indices = " << qkv_output.OffsetToIndices("global_idx") << ";\n"
                            << "let input_offset_idx = output_indices[0] * uniforms.batch_offset + output_indices[1] *"
                            << " uniforms.head_offset + output_indices[2] * uniforms.sequence_offset + output_indices[3];\n";
  if (has_bias_) {
    shader.MainFunctionBody() << "let bias_offset_idx = (input_offset_idx % uniforms.sequence_offset) + uniforms.bias_offset;\n";
  }
  shader.MainFunctionBody() << "qkv_output[global_idx] = qkv_input[input_offset_idx]";
  if (has_bias_) {
    shader.MainFunctionBody() << " + bias[bias_offset_idx];\n";
  } else {
    shader.MainFunctionBody() << ";\n";
  }

  return Status::OK();
}

Status TransferBSDToBNSH(onnxruntime::webgpu::ComputeContext& context, int num_heads, int sequence_length,
                         int head_size, const Tensor* input_tensor, const Tensor* bias, int bias_offset, Tensor* output_tensor) {
  assert(input_tensor->Shape().GetDims().size() == 3);
  assert(output_tensor->Shape().GetDims().size() == 4);

  uint32_t data_size = gsl::narrow<uint32_t>(output_tensor->Shape().Size());
  const int batch_offset = num_heads * sequence_length * head_size;
  const int sequence_offset = num_heads * head_size;
  const int head_offset = head_size;
  bool has_bias = bias != nullptr;

  TransferBSDToBNSHProgram program{has_bias};
  program.AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .SetDispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{data_size},
                            {static_cast<uint32_t>(batch_offset)},
                            {static_cast<uint32_t>(sequence_offset)},
                            {static_cast<uint32_t>(head_offset)},
                            {static_cast<uint32_t>(bias_offset)}});

  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank});
  }

  return context.RunProgram(program);
};

Status AttentionProbsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  if (feed_past_key_) {
    shader.AddInput("past_key", ShaderUsage::UseUniform);
  }
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }

  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  if (has_present_key_) {
    shader.AddOutput("present_key", ShaderUsage::UseUniform);
  }

  shader.AdditionalImplementation() << "var<workgroup> tileQ: array<q_value_t, " << tile_size_ * tile_size_ << ">;\n"
                                    << "var<workgroup> tileK: array<key_value_t, " << tile_size_ * tile_size_ << ">;\n"
                                    << "alias f32_val_t = " << (components_ == 4 ? "vec4<f32>" : (components_ == 2 ? "vec2<f32>" : "f32")) << ";\n";

  shader.MainFunctionBody() << "// x holds the N and y holds the M\n"
                               "let headIdx = workgroup_id.z;\n"
                               "let m = workgroup_id.y * TILE_SIZE;\n"
                               "let n = workgroup_id.x * TILE_SIZE;\n"
                               "let qOffset = uniforms.M * uniforms.K * headIdx + m * uniforms.K;\n";

  if (feed_past_key_ && has_present_key_) {
    shader.MainFunctionBody() << "let kOffset = uniforms.kv_sequence_length * uniforms.K * headIdx;\n"
                              << "let pastKeyOffset = uniforms.past_sequence_length * uniforms.K * headIdx;\n";
  } else {
    shader.MainFunctionBody() << "let kOffset = uniforms.N * uniforms.K * headIdx + n * uniforms.K;\n";
  }

  if (has_present_key_) {
    shader.MainFunctionBody() << "let presentKeyOffset = headIdx * uniforms.N * uniforms.K;\n";
  }

  shader.MainFunctionBody() << "var value = f32_val_t(0);\n"
                               "for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {\n"
                               "  if (global_id.y < uniforms.M && w + local_id.x < uniforms.K) {\n"
                               "    tileQ[TILE_SIZE * local_id.y + local_id.x] = q[qOffset + local_id.y * uniforms.K + w + local_id.x];\n"
                               "  }\n"
                               "  if (n + local_id.y < uniforms.N && w + local_id.x < uniforms.K) {\n"
                               "    var idx = TILE_SIZE * local_id.y + local_id.x;\n";

  if (feed_past_key_ && has_present_key_) {
    shader.MainFunctionBody() << "    if (n + local_id.y < uniforms.past_sequence_length) {\n"
                                 "      tileK[idx] = past_key[pastKeyOffset + (n + local_id.y) * uniforms.K + w + local_id.x];\n"
                                 "    } else {\n"
                                 "      tileK[idx] = key[kOffset + (n + local_id.y - uniforms.past_sequence_length) * uniforms.K + w + local_id.x];\n"
                                 "    }\n";
  } else {
    shader.MainFunctionBody() << "    tileK[idx] = key[kOffset + local_id.y * uniforms.K + w + local_id.x];\n";
  }

  if (has_present_key_) {
    shader.MainFunctionBody() << "    present_key[presentKeyOffset + (n + local_id.y) * uniforms.K + w + local_id.x] = tileK[idx];\n";
  }

  shader.MainFunctionBody() << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "  for (var k: u32 = 0u; k < TILE_SIZE && w+k < uniforms.K; k++) {\n"
                            << "    value += f32_val_t(tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * local_id.x + k]);\n"
                            << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "}\n";

  shader.MainFunctionBody() << "let headOffset = headIdx * uniforms.M * uniforms.N;\n"
                            << "if (global_id.y < uniforms.M && global_id.x < uniforms.N) {\n"
                            << "  let outputIdx = headOffset + global_id.y * uniforms.N + global_id.x;\n"
                            << "  var sum: f32 = " << (components_ == 4 ? "value.x + value.y + value.z + value.w" : (components_ == 2 ? "value.x + value.y" : "value")) << ";\n";

  shader.MainFunctionBody() << "  output[outputIdx] = output_value_t(sum * uniforms.alpha)";
  if (has_attention_bias_) {
    shader.MainFunctionBody() << " + attention_bias[outputIdx]";
  }
  shader.MainFunctionBody() << ";\n"
                            << "}\n";

  return Status::OK();
}

Status ComputeAttentionProbs(onnxruntime::webgpu::ComputeContext& context, int output_count, const Tensor* Q,
                             const Tensor* K, const Tensor* past_key, const Tensor* attention_bias, Tensor* probs, Tensor* present_key,
                             AttentionParameters& parameters, int past_sequence_length, int total_sequence_length) {
  const float alpha = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size))
                                               : parameters.scale;

  const bool feed_past_key = present_key != nullptr && past_key != nullptr && past_key->SizeInBytes() > 0;
  const bool has_present_key = output_count > 1 && past_key;
  const bool has_attention_bias = attention_bias != nullptr;
  constexpr int tile_size = 12;
  const int components = parameters.head_size % 4 == 0 ? 4 : (parameters.head_size % 2 == 0 ? 2 : 1);

  AttentionProbsProgram program{"AttentionProbs", feed_past_key, has_present_key, has_attention_bias, tile_size,
                                components};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {K, ProgramTensorMetadataDependency::TypeAndRank, components}});
  if (feed_past_key) {
    program.AddInput({past_key, ProgramTensorMetadataDependency::TypeAndRank, components});
  }
  if (has_attention_bias) {
    program.AddInput({attention_bias, ProgramTensorMetadataDependency::TypeAndRank});
  }
  program.AddOutputs({{probs, ProgramTensorMetadataDependency::Rank}});
  if (has_present_key) {
    program.AddOutput({present_key, ProgramTensorMetadataDependency::Rank, components});
  }

  const uint32_t vectorized_head_size = parameters.head_size / components;
  program.SetDispatchGroupSize((total_sequence_length + tile_size - 1) / tile_size,
                               (parameters.sequence_length + tile_size - 1) / tile_size,
                               parameters.batch_size * parameters.num_heads)
      .SetWorkgroupSize(tile_size, tile_size)
      .CacheHint(std::to_string(tile_size))
      .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length)},
                            {static_cast<uint32_t>(vectorized_head_size)},
                            {static_cast<uint32_t>(total_sequence_length)},
                            {static_cast<uint32_t>(parameters.num_heads)},
                            {static_cast<float>(alpha)},
                            {static_cast<uint32_t>(past_sequence_length)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length)}})
      .SetOverridableConstants({{static_cast<uint32_t>(tile_size)}});

  return context.RunProgram(program);
}

Status InPlaceSoftmaxProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddOutput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AdditionalImplementation() << "var<workgroup> thread_max: array<f32, " << work_group_size_ << ">;\n"
                                    << "var<workgroup> thread_sum: array<f32, " << work_group_size_ << ">;\n"
                                    << "alias f32_val_t = " << (components_ == 4 ? "vec4<f32>" : (components_ == 2 ? "vec2<f32>" : "f32")) << ";\n";

  shader.MainFunctionBody() << "let local_offset = local_idx * uniforms.elements_per_thread;\n"
                            << "let offset = (global_idx / " << work_group_size_ << ") * uniforms.d_comp + local_offset;\n"
                            << "var thread_max_vector = f32_val_t(-3.402823e+38f);\n"
                            << "for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < uniforms.d_comp; i++) {\n"
                            << "  thread_max_vector = max(f32_val_t(x[offset + i]), thread_max_vector);\n"
                            << "}\n"
                            << "thread_max[local_idx] = " << (components_ == 4 ? "max(max(thread_max_vector.x, thread_max_vector.y), max(thread_max_vector.z, thread_max_vector.w))" : (components_ == 2 ? "max(thread_max_vector.x, thread_max_vector.y)" : "thread_max_vector")) << ";\n"
                            << "workgroupBarrier();\n"
                            << "var max_value =  f32(-3.402823e+38f);\n"
                            << "for (var i = 0u; i < " << work_group_size_ << "; i++) {\n"
                            << "  max_value = max(thread_max[i], max_value);\n"
                            << "}\n"
                            << "var sum_vector = f32_val_t(0);\n"
                            << "for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < uniforms.d_comp; i++) {\n"
                            << "  sum_vector += exp(f32_val_t(x[offset + i]) - max_value);\n"
                            << "}\n"
                            << "thread_sum[local_idx] = " << (components_ == 4 ? "sum_vector.x + sum_vector.y + sum_vector.z + sum_vector.w" : (components_ == 2 ? "sum_vector.x + sum_vector.y" : "sum_vector")) << ";\n"
                            << "workgroupBarrier();\n"
                            << "var sum: f32 = 0;\n"
                            << "for (var i = 0u; i < " << work_group_size_ << "; i++) {\n"
                            << "  sum += thread_sum[i]\n;"
                            << "}\n"
                            << "if (sum == 0) {\n"
                            << "  for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < uniforms.d_comp; i++) {\n"
                            << "    x[offset + i] = x_value_t(x_element_t(uniforms.d_inv));\n"
                            << "  }\n"
                            << "} else {\n"
                            << "  for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < uniforms.d_comp; i++) {\n"
                            << "    var f32input = f32_val_t(x[offset + i]);\n"
                            << "    x[offset + i] = x_value_t(exp(f32input - max_value) / sum);\n"
                            << "  }\n"
                            << "}\n";

  return Status::OK();
}

Status ComputeInPlaceSoftmax(onnxruntime::webgpu::ComputeContext& context, Tensor* probs, int n, int d) {
  const int components = d % 4 == 0 ? 4 : (d % 2 == 0 ? 2 : 1);
  int work_group_size = 64;
  const int d_comp = d / components;
  if (d_comp < work_group_size) {
    work_group_size = 32;
  }
  const int elementsPerThread = (d_comp + work_group_size - 1) / work_group_size;

  InPlaceSoftmaxProgram program{"InPlaceSoftmax", work_group_size, components};
  program.AddOutputs({{probs, ProgramTensorMetadataDependency::TypeAndRank, components}})
      .SetDispatchGroupSize(n)
      .SetWorkgroupSize(work_group_size)
      .AddUniformVariables({{static_cast<float>(1.f / static_cast<float>(d))},
                            {static_cast<uint32_t>(d_comp)},
                            {static_cast<uint32_t>(elementsPerThread)}});

  return context.RunProgram(program);
}

Status VxAttentionScoreProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("probs", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("v", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  if (feed_past_value_) {
    shader.AddInput("past_value", ShaderUsage::UseUniform);
  }

  shader.AddOutput("output", ShaderUsage::UseUniform);
  if (has_present_value_) {
    shader.AddOutput("present_value", ShaderUsage::UseUniform);
  }

  shader.AdditionalImplementation() << "var<workgroup> tileQ: array<probs_value_t, " << tile_size_ * tile_size_ << ">;\n"
                                    << "var<workgroup> tileK: array<v_value_t, " << tile_size_ * tile_size_ << ">;\n";

  shader.MainFunctionBody() << "let headIdx = workgroup_id.z;\n"
                            << "let m = global_id.y;\n"
                            << "let n = global_id.x;\n"
                            << "let offsetA = headIdx * (uniforms.M * uniforms.K) + m * uniforms.K;\n";

  if (feed_past_value_ && has_present_value_) {
    shader.MainFunctionBody() << "let pastValueOffset = headIdx * uniforms.N * uniforms.past_sequence_length + n;\n"
                              << "let vOffset = headIdx * uniforms.N * uniforms.kv_sequence_length + n;\n";
  } else {
    shader.MainFunctionBody() << "let offsetB = headIdx * uniforms.N * uniforms.K + n;\n";
  }

  if (has_present_value_) {
    shader.MainFunctionBody() << "let presentValueOffset = headIdx * uniforms.N * uniforms.K + n;\n";
  }

  shader.MainFunctionBody() << "var value = probs_element_t(0);\n"
                            << "for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {\n"
                            << "  if (m < uniforms.M && w + local_id.x < uniforms.K) {\n"
                            << "    tileQ[TILE_SIZE * local_id.y + local_id.x] = probs[offsetA + w + local_id.x];\n"
                            << "  }\n"
                            << "  if (n < uniforms.N && w + local_id.y < uniforms.K) {\n"
                            << "    var idx = TILE_SIZE * local_id.y + local_id.x;\n";

  if (feed_past_value_ && has_present_value_) {
    shader.MainFunctionBody() << "    if (w + local_id.y < uniforms.past_sequence_length) {\n"
                              << "      tileK[idx] = past_value[pastValueOffset + (w + local_id.y) * uniforms.N];\n"
                              << "    } else {\n"
                              << "      tileK[idx] = v[vOffset + (w + local_id.y - uniforms.past_sequence_length) * uniforms.N];\n"
                              << "    }\n";
  } else {
    shader.MainFunctionBody() << "    tileK[idx] = v[offsetB + (w + local_id.y) * uniforms.N];\n";
  }

  if (has_present_value_) {
    shader.MainFunctionBody() << "    present_value[presentValueOffset + (w + local_id.y) * uniforms.N] = tileK[idx];\n";
  }

  shader.MainFunctionBody() << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "  for (var k: u32 = 0u; k < TILE_SIZE && w+k < uniforms.K; k++) {\n"
                            << "    value += tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * k + local_id.x];\n"
                            << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "}\n";

  shader.MainFunctionBody() << "// we need to transpose output from BNSH_v to BSND_v\n"
                            << "let batchIdx = workgroup_id.z / uniforms.num_heads;\n"
                            << "let currentBatchHeadNumber = workgroup_id.z % uniforms.num_heads;\n"
                            << "if (m < uniforms.M && n < uniforms.N) {\n"
                            << "  let outputIdx = batchIdx * uniforms.M * uniforms.v_hidden_size + "
                            << "  m * uniforms.v_hidden_size + currentBatchHeadNumber * uniforms.N + n;\n"
                            << "  output[outputIdx] = value;\n"
                            << "}\n";

  return Status::OK();
}

Status ComputeVxAttentionScore(onnxruntime::webgpu::ComputeContext& context, int output_count,
                               const Tensor* probs,
                               const Tensor* V,
                               const Tensor* past_value,
                               Tensor* output,
                               Tensor* present_value,
                               AttentionParameters& parameters,
                               int past_sequence_length,
                               int total_sequence_length) {
  const bool feed_past_value = present_value != nullptr && past_value != nullptr && past_value->SizeInBytes() > 0;
  const bool has_present_value = output_count > 1 && past_value != nullptr;
  constexpr int tile_size = 12;

  VxAttentionScoreProgram program{"VxAttentionScore", feed_past_value, has_present_value, tile_size};
  program.AddInputs({{probs, ProgramTensorMetadataDependency::TypeAndRank},
                     {V, ProgramTensorMetadataDependency::TypeAndRank}});
  if (feed_past_value) {
    program.AddInput({past_value, ProgramTensorMetadataDependency::TypeAndRank});
  }
  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank}});
  if (has_present_value) {
    program.AddOutput({present_value, ProgramTensorMetadataDependency::TypeAndRank});
  }

  program.SetDispatchGroupSize((parameters.v_head_size + tile_size - 1) / tile_size,
                               (parameters.sequence_length + tile_size - 1) / tile_size,
                               parameters.batch_size * parameters.num_heads)
      .SetWorkgroupSize(tile_size, tile_size)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length)},
                            {static_cast<uint32_t>(total_sequence_length)},
                            {static_cast<uint32_t>(parameters.v_head_size)},
                            {static_cast<uint32_t>(parameters.num_heads)},
                            {static_cast<uint32_t>(parameters.v_hidden_size)},
                            {static_cast<uint32_t>(past_sequence_length)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length)}})
      .SetOverridableConstants({{static_cast<uint32_t>(tile_size)}});
  ;

  return context.RunProgram(program);
}

Status ApplyAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                      const Tensor* past_key, const Tensor* past_value, Tensor* output, Tensor* present_key, Tensor* present_value,
                      AttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  const int output_count = std::min({context.OutputCount(), 1 + (past_key != nullptr ? 1 : 0) + (past_value != nullptr ? 1 : 0)});
  const int past_sequence_length = output_count > 1 ? parameters.past_sequence_length : 0;
  const int total_sequence_length = past_sequence_length + parameters.kv_sequence_length;

  const TensorShapeVector probs_dims({parameters.batch_size, parameters.num_heads,
                                      parameters.sequence_length, total_sequence_length});
  const TensorShape probs_shape(probs_dims);
  Tensor probs = context.CreateGPUTensor(Q->DataType(), probs_shape);
  ORT_RETURN_IF_ERROR(ComputeAttentionProbs(context, output_count, Q, K, past_key, attention_bias, &probs, present_key,
                                            parameters, past_sequence_length, total_sequence_length));

  ORT_RETURN_IF_ERROR(ComputeInPlaceSoftmax(context, &probs,
                                            parameters.batch_size * parameters.num_heads * parameters.sequence_length, total_sequence_length));

  ORT_RETURN_IF_ERROR(ComputeVxAttentionScore(context, output_count, &probs, V, past_value, output, present_value,
                                              parameters, past_sequence_length, total_sequence_length));

  return Status::OK();
}

Status CopyKVCacheProgram::GenerateShaderCode(ShaderHelper& shader) const {
   // Expectations are
   //    qkv have same number of heads and hidden dimension (head size).
   //    qkv are in BSNH format.
   //            B - batch size but shader only supports batch_size 1.
   //            S - current sequence length but shader supports only S = 1.
   //            N - number of heads.
   //            H - head size or hidden dimension for each qkv head.
   //  KV cache is stored as BN(total_sequence_length)H
   //  Attention bias is in BN(total_sequence_length)
   shader.AddInput("key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
   shader.AddInput("value", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
   if (has_past_) {
     shader.AddInput("past_key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
     shader.AddInput("past_value", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
   }
   shader.AddOutput("present_key", ShaderUsage::UseUniform);
   shader.AddOutput("present_value", ShaderUsage::UseUniform);

  shader.MainFunctionBody() << "let headIdx = workgroup_id.z;\n"
                << "let kIdx = workgroup_id.x;\n"
                << "let presentKeyOffset = headIdx * num_workgroups.x * uniforms.vectorized_head_size + (kIdx)*uniforms.vectorized_head_size;\n";
  if (has_past_) {
    shader.MainFunctionBody() << "if (kIdx < uniforms.past_sequence_length) {\n"
                              << "  let pastKeyOffset = headIdx * uniforms.past_sequence_length * uniforms.vectorized_head_size + (kIdx)*uniforms.vectorized_head_size;\n"
                              << "  for (var w: u32 = 0u; w < uniforms.vectorized_head_size; w ++) {\n"
                              << "    present_key[presentKeyOffset+w] = past_key[pastKeyOffset+w];\n"
                              << "    present_value[presentKeyOffset+w] = past_value[pastKeyOffset+w];\n"
                              << "  }\n"
                              << "}\n"
                              << "else if (kIdx >= uniforms.past_sequence_length) {\n";
  } else {
    shader.MainFunctionBody() << "if (kIdx >= uniforms.past_sequence_length) {\n";
  }
  shader.MainFunctionBody() << "  let nkIdx = kIdx - uniforms.past_sequence_length;\n"
                << "  // Assumes kv have BSNH layout. num_workgroups.z is the num_head as per the dispatch requirement.\n"
                << "  let nOffset = nkIdx * uniforms.vectorized_head_size * num_workgroups.z + headIdx*uniforms.vectorized_head_size;\n"
                << "  // Assumes kv have BNSH layout.\n"
                << "  // let nOffset = headIdx * uniforms.kv_sequence_length * uniforms.vectorized_head_size + nkIdx * uniforms.vectorized_head_size;\n"
                << "  for (var w: u32 = 0u; w < uniforms.vectorized_head_size; w ++) {\n"
                << "    present_key[presentKeyOffset+w] = key[nOffset+w];\n"
                << "    present_value[presentKeyOffset+w] = value[nOffset+w];\n"
                << "  }\n"
                << "}\n";

   return Status::OK();
}

Status CopyKVCache(onnxruntime::webgpu::ComputeContext& context, AttentionParameters& parameters,
                             const Tensor* K, const Tensor* past_key, Tensor* present_key,
                             const Tensor* V, const Tensor* past_value, Tensor* present_value,
                             int past_sequence_length, int total_sequence_length) {

  const int components = parameters.head_size % 4 == 0 ? 4 : (parameters.head_size % 2 == 0 ? 2 : 1);
  bool has_past = (past_sequence_length != 0);
  CopyKVCacheProgram program{"CopyKVCache", components, has_past};
  if (has_past) {
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {past_key, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {past_value, ProgramTensorMetadataDependency::TypeAndRank, components}});
  } else {
    program.AddInputs({{K, ProgramTensorMetadataDependency::TypeAndRank, components},
                       {V, ProgramTensorMetadataDependency::TypeAndRank, components}});
  }

  program.AddOutputs({{present_key, ProgramTensorMetadataDependency::Rank, components},
                      {present_value, ProgramTensorMetadataDependency::Rank, components}});

  program.SetDispatchGroupSize(total_sequence_length, 1, parameters.num_heads)
      .SetWorkgroupSize(1)
      .CacheHint(std::to_string(components) + std::to_string(has_past))
      .AddUniformVariables({{static_cast<uint32_t>(past_sequence_length)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length)},
                            {static_cast<uint32_t>(parameters.head_size/ components)}});

  return context.RunProgram(program);
}

Status FlashAttentionProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Expectations are
  //    qkv have same number of heads and hidden dimension (head size).
  //    qkv are in BSNH format.
  //            B - batch size but shader only supports batch_size 1.
  //            S - current sequence length but shader supports only S = 1.
  //            N - number of heads.
  //            H - head size or hidden dimension for each qkv head.
  //  KV cache is stored as BN(total_sequence_length)H
  //  Attention bias is in BN(new_sequence_length)(total_sequence_length)
  //
  //  Expectation is that present_key, and present_value contain past key and values since
  //  we are out of storage buffers a shader can have and both past/present cant be passed.
  // The hidden size of each q head should be a multiple of 4 because shader uses vectorized loads.
  constexpr int vectorization_size = 4;
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("present_key", ShaderUsage::UseUniform);
  shader.AddInput("present_value", ShaderUsage::UseUniform);
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);

  // SUBGROUP_SIZE has to be the same as sg_size. For intel this will be 8.
  // TILE_SIZE is the number of groups sharing the k_tile.
  // TILE_SIZE has to be <= SUBGROUP_SIZE. Ideal perf of computeSoftMax is when
  // TILE_SIZE == SUBGROUP_SIZE. This is a sperate constant from SUBGROUP_SIZE
  // because SUBGROUP_SIZE * TILE_SIZE has to be <= 256 as per webgpu
  // gpu limits. For Intel this TILE_SIZE will be 8.
  shader.AdditionalImplementation() << "const SUBGROUP_SIZE: u32 = " << subgroup_size_ << ";\n"
                                    << "const TILE_SIZE: u32 = " << tile_size_ << ";\n"
                                    << "const VECTOR_SIZE: u32 = " << vectorization_size << ";\n"
                                    << "const QKV_HEAD_SIZE: u32 = " << qkv_head_size_ << ";\n"
                                    << "const QKV_HEAD_VECTORIZED_SIZE: u32 = QKV_HEAD_SIZE / VECTOR_SIZE;\n"
                                    << "const NUM_HEADS: u32 = " << qkv_num_heads_ << ";\n"
                                    << "const MIN_VALUE : q_element_t = -6504.0h;\n";

  // Best to keep SHM usage per workgroup < 8KB. 4KB is the limit on a 48EU tigerlake
  // GPU afterwhich workgroups will be unscheduled to make space for memory.
  shader.AdditionalImplementation() << "var<workgroup> q_tile : array<array<q_value_t, QKV_HEAD_VECTORIZED_SIZE>, TILE_SIZE>; // 96 * 8 * 2 = 1.5KB.\n"
                                    << "var<workgroup> k_tile : array<array<q_value_t, QKV_HEAD_VECTORIZED_SIZE>, TILE_SIZE>; // 96 * 8 * 2 = 1.5KB.\n"
                                    << "var<workgroup> v_tile : array<array<q_value_t, QKV_HEAD_VECTORIZED_SIZE>, TILE_SIZE>; // 96 * 8 * 2 = 1.5KB.\n"
                                    << "var<workgroup> o_tile : array<array<q_value_t, QKV_HEAD_VECTORIZED_SIZE>, TILE_SIZE>; // 96 * 8 * 2 = 1.5KB.\n"
                                    << "var<workgroup> qk_tile : array<array<q_element_t, TILE_SIZE>, TILE_SIZE>; // 8 * 2 * 8 = 128\n"
                                    << "var<workgroup> max_tile : array<q_element_t, TILE_SIZE>; // 2 * 8 = 16\n"
                                    << "var<workgroup> denom_tile : array<q_element_t, TILE_SIZE>; // 2 * 8 = 16\n"
                                    << "var<workgroup> o_ratio : array<q_element_t, TILE_SIZE>; // 2 * 8 = 16\n";

  shader.AdditionalImplementation() << R"HELPER_FN(

fn loadq(slot: u32, q_idx_global : u32, head_idx: u32, sg_id : u32, sg_size : u32)
{
    // Stored as float16[batch_size,sequence_length,3072] the inputs as per onnx MHA
    // This is the layout if TransferBSDToBNSH has not been run.
    let offset = q_idx_global * (QKV_HEAD_VECTORIZED_SIZE) * NUM_HEADS + QKV_HEAD_VECTORIZED_SIZE * head_idx;
    // Stored as BNSH - which is what webgpu uses after TransferBSDToBNSH has been run.
    // let offset = head_idx * uniforms.new_sequence_length * QKV_HEAD_VECTORIZED_SIZE + q_idx_global * QKV_HEAD_VECTORIZED_SIZE;
    for (var idx:u32 = sg_id; idx < QKV_HEAD_VECTORIZED_SIZE; idx+= sg_size)
    {
        var value = q[idx+offset];
        q_tile[slot][idx] = value;
    }
}

fn loadk(slot: u32, k_idx_global : u32, head_idx: u32, sg_id: u32, sg_size: u32)
{
    // Stored as float16[batch_size,num_heads,present_sequence_length,96]
    let offset = head_idx * uniforms.present_sequence_length * QKV_HEAD_VECTORIZED_SIZE + k_idx_global * QKV_HEAD_VECTORIZED_SIZE;
    for (var idx:u32 = sg_id; idx < QKV_HEAD_VECTORIZED_SIZE; idx+=sg_size)
    {
        var value = present_key[idx+offset];
        k_tile[slot][idx] = value;
    }
}

fn loadv(slot: u32, v_idx_global : u32, head_idx: u32, sg_id: u32, sg_size: u32)
{
    // Stored as float16[batch_size,num_heads,present_sequence_length,96]
    let offset = head_idx * uniforms.present_sequence_length * QKV_HEAD_VECTORIZED_SIZE + v_idx_global * QKV_HEAD_VECTORIZED_SIZE;
    for (var idx:u32 = sg_id; idx < QKV_HEAD_VECTORIZED_SIZE; idx+=sg_size)
    {
        v_tile[slot][idx] = present_value[idx+offset];
    }
}

fn loadAttentionBias(q_row: u32, q_idx_global : u32, k_col: u32, k_idx_global : u32, head_idx: u32)
{
    // Stored as float16[batch_size,num_heads,new_seq_length,total_sequence_length]
    if (q_idx_global >= uniforms.new_sequence_length  || k_idx_global >= uniforms.present_sequence_length || k_col >= TILE_SIZE) {
        qk_tile[q_row][k_col] = 0.0;
        return;
    }
    let offset = head_idx * uniforms.new_sequence_length * uniforms.present_sequence_length + q_idx_global * uniforms.present_sequence_length + k_idx_global;
    qk_tile[q_row][k_col] = attention_bias[offset];
}

fn writeo(slot: u32, o_idx_global : u32, head_idx: u32, sg_id : u32, sg_size : u32)
{
    // Stored as float16[batch_size,sequence_length,3072]
    let offset = o_idx_global * NUM_HEADS * QKV_HEAD_VECTORIZED_SIZE + head_idx * QKV_HEAD_VECTORIZED_SIZE;
    for (var idx:u32 = sg_id; idx < QKV_HEAD_VECTORIZED_SIZE; idx += sg_size)
    {
        let value = o_tile[slot][idx];
        output[offset+idx] = value;
    }
}

fn computeDotProduct(q_idx: u32, k_idx: u32, sg_id: u32, sg_size : u32)
{
    var sum:vec4<q_element_t> = q_value_t(0, 0, 0, 0);
    // idx is not initialized to sg_id to ensure uniformity because the loop uses
    // subgroupAdd and unused lanes need to be initialized with 0 for correctness.
    for (var idx:u32 = 0; idx < QKV_HEAD_VECTORIZED_SIZE; idx+= sg_size)
    {
        var result = q_value_t(0);
        let sg_idx = idx+sg_id;
        // QKV_HEAD_VECTORIZED_SIZE is divisible by the subgroup size this if check is not
        // required. Hopefully the compiler sees the first half of this if statement and
        // removes this if instruction.
        if (QKV_HEAD_VECTORIZED_SIZE % sg_size == 0 || sg_idx < QKV_HEAD_VECTORIZED_SIZE)
        {
            result = q_tile[q_idx][sg_idx]*k_tile[k_idx][sg_idx];
        }
        sum += subgroupAdd(result);
    }

    if (sg_id == 0)
    {
        let single_sum : q_element_t = sum.x + sum.y + sum.z + sum.w;
        let sqrt_dk = q_element_t(uniforms.alpha);
        let value = single_sum * sqrt_dk;
        qk_tile[q_idx][k_idx] += value;
    }
}

fn computeSoftMax(q_idx: u32, sg_id:u32, enabled:bool)
{
    var x = MIN_VALUE;
    if (enabled){
        x = qk_tile[q_idx][sg_id];
    }
    var max_value = subgroupMax(x);
    max_value = max(max_tile[q_idx], max_value);
    let sub = x - max_value;
    var value:q_element_t = 0;
    if (enabled) {
        value = exp(sub);
    }
    let sum = subgroupAdd(value);

    // Compute lhs term of update di prime and the compute di prime.
    let dleft = denom_tile[q_idx] * exp(max_tile[q_idx]-max_value);
    var d = dleft + sum;
    if (d == 0)
    {
        // Avoid division by zero by setting d to a really small value.
        d = 0.0000001h;
    }
    qk_tile[q_idx][sg_id] = value / d;
    if (sg_id == 0)
    {
        max_tile[q_idx] = max_value;
        denom_tile[q_idx] = d;
        o_ratio[q_idx] = dleft / d;
    }
}

fn computeO(q_idx: u32, sg_id:u32, enabled:bool)
{
    var attn = q_element_t(0);
    if (enabled)
    {
      attn = qk_tile[q_idx][sg_id];
    }
    for (var i:u32 = 0; i < QKV_HEAD_VECTORIZED_SIZE; i++)
    {
        let val = v_tile[sg_id][i];
        var intermediate = attn * val;
        let sum = subgroupAdd(intermediate);
        if (sg_id == 0)
        {
            let o_ratio = o_ratio[q_idx];
            let old_o = o_tile[q_idx][i];
            let new_o = ( o_ratio * old_o) +  sum;
            o_tile[q_idx][i] = new_o;
        }
    }
}

)HELPER_FN";

// Shader is designed to be dispatched as Dispatch(num_heads, new_sequence_length / TILE_SIZE, 1)
// Each workgroup is responsible for a range of q values (TILE_SIZE) and visits all Ks for those q's.
  shader.MainFunctionBody() << R"MAIN_FN(
let head_idx = workgroup_id.x;

let wave_x = u32(local_id.x / 4);
let wave_y = u32(local_id.y / 4);
// It is always the case that 0 <= wave_id < TILE_SIZE
let wave_id:u32 = wave_x + wave_y * 4;

let q_idx_start = workgroup_id.y * TILE_SIZE;
let q_idx_global = q_idx_start + wave_id;
let q_idx_global_using_wave_valid = q_idx_global < uniforms.new_sequence_length;
if (q_idx_global_using_wave_valid)
{
  // Each invocation (wave_id) gets lane threads (subgroup threads) and is responsible for 1 query.
  loadq(wave_id, q_idx_global, head_idx, sg_id, sg_size);
}
if (sg_id == 0)
{
  max_tile[wave_id] = MIN_VALUE;
}

for(var k_start = 0u; k_start < uniforms.present_sequence_length; k_start+=TILE_SIZE)
{
    let k_idx_global = k_start+wave_id;
    let k_idx_global_using_wave_valid = k_idx_global < uniforms.present_sequence_length;
    if (k_idx_global_using_wave_valid) {
        // Leveraging the subgroup lanes for parallelism, load into slot wave_id
        // K/V values from k_start+wave_id.
        loadk(wave_id, k_idx_global, head_idx, sg_id, sg_size);
        loadv(wave_id, k_idx_global, head_idx, sg_id, sg_size);
        // Next, we want for every q row (wave_id) to populate bias for new sequence length
        // (k_start+sg_id). loadAttentionBias handles range checking q_idx_global,
        // and sg_id, (k_start+sg_id).
        loadAttentionBias(wave_id, q_idx_global, sg_id, k_start+sg_id, head_idx);
    }
    workgroupBarrier();
    if (k_idx_global_using_wave_valid)
    {
      for (var q_idx = 0u; q_idx < TILE_SIZE && q_idx_start + q_idx < uniforms.new_sequence_length; q_idx++)
      {
          // Leveraging the subgroups for parallelism, compute dot product of QK.
          // Because for the case of new_seq 1, there is a single query and context length of K
          // we iterate over q and use the waves for K so that this step can use all the waves in
          // in the workgroup.
          // We validate q_idx,wave_id to be less than TILE_SIZE, computeDotProduct only needs to
          // validate sg_id as being less than QKV_HEAD_VECTORIZED_SIZE.
          computeDotProduct(q_idx, wave_id, sg_id, sg_size);
      }
    }
    let wave_lane_valid:bool = q_idx_global_using_wave_valid && sg_id < TILE_SIZE && sg_id + k_start < uniforms.present_sequence_length;
    computeSoftMax(wave_id, sg_id, wave_lane_valid);
    computeO(wave_id, sg_id, wave_lane_valid);
}
workgroupBarrier();
if (q_idx_global_using_wave_valid)
{
  writeo(wave_id, q_idx_global, head_idx, sg_id, sg_size);
}
)MAIN_FN";

  return Status::OK();
}

Status ApplyFlashAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                      Tensor* output, const Tensor* past_key, Tensor* present_key, const Tensor* past_value, Tensor* present_value,
                      AttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context) {
  ORT_RETURN_IF_ERROR(CopyKVCache(context, parameters, K, past_key, present_key, V, past_value, present_value, parameters.past_sequence_length, parameters.total_sequence_length));

  constexpr int subgroup_size = 16;
  constexpr int tile_size = 16;
  bool has_attention_bias = attention_bias != nullptr;
  FlashAttentionProgram program{"FlashAttention", has_attention_bias, subgroup_size, tile_size, parameters.head_size, parameters.num_heads};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, 4},
                     {present_key, ProgramTensorMetadataDependency::TypeAndRank, 4},
                     {present_value, ProgramTensorMetadataDependency::TypeAndRank, 4},
                     {attention_bias, ProgramTensorMetadataDependency::TypeAndRank}});
  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank, 4}});
  const float alpha = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size))
                                               : parameters.scale;
  std::string cache_hint = std::to_string(has_attention_bias) +
    std::to_string(subgroup_size) +
    std::to_string(tile_size) +
    std::to_string(parameters.head_size) +
    std::to_string(parameters.num_heads);
  program.SetDispatchGroupSize(parameters.num_heads, (parameters.sequence_length + tile_size - 1) / tile_size, 1)
    .SetWorkgroupSize(subgroup_size, subgroup_size)
    .CacheHint(cache_hint)
    .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length)},
                          {static_cast<uint32_t>(parameters.total_sequence_length)},
                          {alpha}});

  return context.RunProgram(program);
}

MultiHeadAttention::MultiHeadAttention(const OpKernelInfo& info)
    : WebGpuKernel(info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);
  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  is_unidirectional_ = info.GetAttrOrDefault<int64_t>("unidirectional", 0) == 1;
  ORT_ENFORCE(!is_unidirectional_, "Unidirectional MHA does not support webgpu kernel");
}

Status MultiHeadAttention::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* query = context.Input(0);
  const Tensor* key = context.Input(1);
  const Tensor* value = context.Input(2);
  const Tensor* bias = context.Input(3);
  const Tensor* key_padding_mask = context.Input(4);
  const Tensor* attention_bias = context.Input(5);
  const Tensor* past_key = context.Input(6);
  const Tensor* past_value = context.Input(7);

  if (query->Shape().GetDims().size() == 5) {
    ORT_NOT_IMPLEMENTED("Packed QKV of shape (B, L, N, 3, H) not implemented for webgpu");
  }
  if (key != nullptr && key->Shape().GetDims().size() == 5) {
    ORT_NOT_IMPLEMENTED("Packed KV not implemented for webgpu");
  }
  if (key_padding_mask) {
    ORT_NOT_IMPLEMENTED("input `key_padding_mask` not implemented for webgpu");
  }

  AttentionParameters parameters;
  ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckInputs<Tensor>(query, key, value,
                                                                      bias, key_padding_mask, attention_bias, past_key, past_value, nullptr, &parameters,
                                                                      num_heads_, mask_filter_value_, scale_, is_unidirectional_, false, kMultiHeadAttention,
                                                                      context.DeviceLimits().maxComputeInvocationsPerWorkgroup));

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context.Output(0, output_shape);

  // If optional outputs aren't needed, present_key and present_value will be null
  std::vector<int64_t> present_dims{
      parameters.batch_size,
      parameters.num_heads,
      parameters.total_sequence_length,
      parameters.head_size,
  };
  TensorShape present_shape(present_dims);
  Tensor* present_key = context.Output(1, present_shape);
  Tensor* present_value = context.Output(2, present_shape);

  if (parameters.batch_size == 1 &&
    bias == nullptr &&
    past_key != nullptr && past_value != nullptr && past_key->SizeInBytes() > 0 &&
    present_key != nullptr && present_value != nullptr && present_key->SizeInBytes() > 0 &&
    present_value->SizeInBytes() > 0 && parameters.head_size % 4 == 0) {
    return ApplyFlashAttention(query, key, value, attention_bias, output, past_key, present_key, past_value,
                          present_value, parameters, context);
  }

  TensorShapeVector q_new_dims({parameters.batch_size, parameters.num_heads,
                                parameters.sequence_length, parameters.head_size});
  TensorShape q_new_shape(q_new_dims);
  Tensor Q = context.CreateGPUTensor(query->DataType(), q_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(
      context, parameters.num_heads, parameters.sequence_length, parameters.head_size, query, bias, 0, &Q));

  if (parameters.qkv_format == Q_K_V_BSNH_BNSH_BNSH) {  // key and value in BNSH format
    return ApplyAttention(&Q, key, value, attention_bias, past_key, past_value, output, present_key,
                          present_value, parameters, context);
  }

  TensorShapeVector k_new_dims({parameters.batch_size, parameters.num_heads,
                                parameters.kv_sequence_length, parameters.head_size});
  TensorShape k_new_shape(k_new_dims);
  Tensor K = context.CreateGPUTensor(key->DataType(), k_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.num_heads, parameters.kv_sequence_length,
                                        parameters.head_size, key, bias, parameters.hidden_size, &K));

  TensorShapeVector v_new_dims({parameters.batch_size, parameters.num_heads,
                                parameters.kv_sequence_length, parameters.v_head_size});
  TensorShape v_new_shape(v_new_dims);
  Tensor V = context.CreateGPUTensor(value->DataType(), v_new_shape);
  ORT_RETURN_IF_ERROR(TransferBSDToBNSH(context, parameters.num_heads, parameters.kv_sequence_length,
                                        parameters.v_head_size, value, bias, 2 * parameters.hidden_size, &V));

  // Compute the attention score and apply the score to V
  return ApplyAttention(&Q, &K, &V, attention_bias, past_key, past_value, output, present_key,
                        present_value, parameters, context);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
