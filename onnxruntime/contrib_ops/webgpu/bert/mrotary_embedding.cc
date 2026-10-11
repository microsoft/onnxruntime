// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/mrotary_embedding.h"

#include <array>
#include <limits>

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace mrotary_embedding_helper;

ONNX_OPERATOR_KERNEL_EX(
    MRotaryEmbedding,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()),
    MRotaryEmbedding);

Status MRotaryEmbeddingProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
  const auto& position_ids = shader.AddInput("position_ids", ShaderUsage::None);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::None);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::None);
  const auto& output = shader.AddOutput("output", ShaderUsage::None);

  return WGSL_TEMPLATE_APPLY(shader, "bert/mrotary_embedding.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(interleaved, interleaved_),
                             WGSL_TEMPLATE_PARAMETER(position_ids_use_storage_type, true),
                             WGSL_TEMPLATE_PARAMETER(sectioned, mrope_layout_ == MRopeLayout::kSectioned),
                             WGSL_TEMPLATE_PARAMETER(transposed, transposed_),
                             WGSL_TEMPLATE_VARIABLE(cos_cache, cos_cache),
                             WGSL_TEMPLATE_VARIABLE(input, input),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(position_ids, position_ids),
                             WGSL_TEMPLATE_VARIABLE(sin_cache, sin_cache));
}

MRotaryEmbedding::MRotaryEmbedding(const OpKernelInfo& info) : WebGpuKernel(info) {
  scale_ = info.GetAttrOrDefault<float>("scale", 1.0f);
  const int64_t rotary_embedding_dim = info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0);
  const int64_t num_heads = info.GetAttrOrDefault<int64_t>("num_heads", 0);
  ORT_ENFORCE(rotary_embedding_dim >= 0 && rotary_embedding_dim <= std::numeric_limits<int>::max(),
              "rotary_embedding_dim must be in range [0, ", std::numeric_limits<int>::max(),
              "]. Actual value: ", rotary_embedding_dim);
  ORT_ENFORCE(num_heads >= 0 && num_heads <= std::numeric_limits<int>::max(),
              "num_heads must be in range [0, ", std::numeric_limits<int>::max(),
              "]. Actual value: ", num_heads);
  rotary_embedding_dim_ = static_cast<int>(rotary_embedding_dim);
  num_heads_ = static_cast<int>(num_heads);
  interleaved_ = info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1;
  is_packed_batching_ = info.GetAttrOrDefault<int64_t>("is_packed_batching", 0) == 1;
  mrope_layout_ = info.GetAttrOrDefault<int64_t>("mrope_layout", 0);
  ORT_ENFORCE(info.GetAttrs<int64_t>("mrope_section", mrope_section_).IsOK(),
              "MRotaryEmbedding: 'mrope_section' attribute is required");

  if (rotary_embedding_dim_ > 0) {
    ORT_ENFORCE(num_heads_ > 0, "num_heads must be provided if rotary_embedding_dim is specified");
  }
}

Status MRotaryEmbedding::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* input = context.Input<Tensor>(0);
  const auto* position_ids = context.Input<Tensor>(1);
  const auto* cos_cache = context.Input<Tensor>(2);
  const auto* sin_cache = context.Input<Tensor>(3);

  MRotaryParameters parameters{};
  ORT_RETURN_IF_ERROR(CheckInputs<Tensor>(input, position_ids, cos_cache, sin_cache,
                                          num_heads_, rotary_embedding_dim_, mrope_section_,
                                          mrope_layout_, &parameters));

  auto* output = context.Output(0, input->Shape());
  if (input->Shape().Size() == 0) {
    return Status::OK();
  }

  if (!is_packed_batching_ && parameters.sequence_length > parameters.max_sequence_length) {
    ORT_NOT_IMPLEMENTED("Updating cos_cache and sin_cache in MRotaryEmbedding is not currently supported");
  }

  const uint32_t output_size = onnxruntime::narrow<uint32_t>(input->Shape().Size());
  const std::array<uint32_t, 3> mrope_section{
      static_cast<uint32_t>(parameters.mrope_section[0]),
      static_cast<uint32_t>(parameters.mrope_section[1]),
      static_cast<uint32_t>(parameters.mrope_section[2])};

  MRotaryEmbeddingProgram program{interleaved_, parameters.transposed, parameters.mrope_layout};
  program.CacheHint(interleaved_, parameters.transposed, static_cast<int>(parameters.mrope_layout))
      .AddInputs({{input, ProgramTensorMetadataDependency::TypeAndRank},
                  {position_ids, ProgramTensorMetadataDependency::TypeAndRank},
                  {cos_cache, ProgramTensorMetadataDependency::TypeAndRank},
                  {sin_cache, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{scale_},
                            {output_size},
                            {static_cast<uint32_t>(parameters.batch_size)},
                            {static_cast<uint32_t>(parameters.sequence_length)},
                            {static_cast<uint32_t>(parameters.num_heads)},
                            {static_cast<uint32_t>(parameters.head_size)},
                            {static_cast<uint32_t>(parameters.rotary_embedding_dim)},
                            {static_cast<uint32_t>(parameters.max_sequence_length)},
                            {gsl::make_span(mrope_section)}});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
