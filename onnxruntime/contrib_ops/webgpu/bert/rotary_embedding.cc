// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/bert/rotary_embedding.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    RotaryEmbedding,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()),
    RotaryEmbedding);

Status RotaryEmbeddingProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform);
  const auto& position_ids = shader.AddInput("position_ids", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
  // TODO: remove output_indices.
  const auto& output_indices = shader.AddIndices("output_indices", false);
  const auto interleaved_str = interleaved_ ? "true" : "false";
  shader.MainFunctionBody() << "  let half_rotary_emb_dim = uniforms.cos_cache_shape[1];\n"
                               "  let bsnh = global_idx / uniforms.global_stride % uniforms.global_shape;\n"
                               "  let size = uniforms.global_shape[0] * uniforms.global_stride[0];\n"
                               "  if (global_idx >= size) { return; }\n"
                               "  if (bsnh[3] < half_rotary_emb_dim) {\n"
                            << "    let position_ids_idx = " << position_ids.BroadcastedIndicesToOffset("bsnh.xy", output_indices) << ";\n"
                            << "    let position_id = u32(" << position_ids.GetByOffset("position_ids_idx") << ") + select(0, bsnh[1], position_ids_idx == 0);\n"
                            << "    let i = dot(bsnh, uniforms.input_output_stride) + select(0, bsnh[3], " << interleaved_str << ");\n"
                            << "    let j = i + select(half_rotary_emb_dim, 1, " << interleaved_str << ");\n"
                            << "    let re = " << input.GetByOffset("i") << " * " << cos_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << " - " << input.GetByOffset("j") << " * " << sin_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << ";\n"
                            << "    " << output.SetByOffset("i", "re") << "\n"
                            << "    let im = " << input.GetByOffset("i") << " * " << sin_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << " + " << input.GetByOffset("j") + " * " << cos_cache.GetByIndices("vec2<u32>(position_id, bsnh[3])") << ";\n"
                            << "    " << output.SetByOffset("j", "im") << "\n"
                            << "  } else { \n"
                               "    let k = dot(bsnh, uniforms.input_output_stride) + half_rotary_emb_dim;\n"
                            << "    " << output.SetByOffset("k", input.GetByOffset("k")) << "\n"
                            << "  }";

  return Status::OK();
}

RotaryEmbedding::RotaryEmbedding(const OpKernelInfo& info) : WebGpuKernel(info) {
  scale_ = info.GetAttrOrDefault<float>("scale", 1.0);
  rotary_embedding_dim_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0));
  num_heads_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("num_heads", 0));
  interleaved_ = (info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1);
  is_packed_batching_ = (info.GetAttrOrDefault<int64_t>("is_packed_batching", 0) == 1);
}

Status RotaryEmbedding::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* input = context.Input<Tensor>(0);
  const auto input_shape = input->Shape();
  const auto* position_ids = context.Input<Tensor>(1);
  const auto* cos_cache = context.Input<Tensor>(2);
  const auto* sin_cache = context.Input<Tensor>(3);
  auto* output = context.Output(0, input_shape);

  const auto batch_size = gsl::narrow<uint32_t>(input->Shape()[0]);
  const auto batch_stride = gsl::narrow<uint32_t>(input_shape.SizeFromDimension(1));
  const auto sequence_length = gsl::narrow<uint32_t>(input_shape[input_shape.NumDimensions() - 2]);
  const auto hidden_size = batch_stride / sequence_length;
  const auto half_rotary_embedding_dim = gsl::narrow<uint32_t>(cos_cache->Shape()[1]);
  const auto head_size = rotary_embedding_dim_ == 0 ? half_rotary_embedding_dim * 2 : hidden_size / num_heads_;

  // Rotary embeddings will be calculated in a pair-wise fashion. In accordance, use the shape
  // [batch size, sequence length, num of heads, num of pairs to rotate + num of dims to copy]
  // to unfold the global index in shader.
  const TensorShape global_shape({batch_size,
                                  sequence_length,
                                  hidden_size / head_size,
                                  head_size - half_rotary_embedding_dim});

  const auto rank = global_shape.NumDimensions();
  std::vector<uint32_t> global_dims(rank);
  std::vector<uint32_t> global_strides(rank);
  for (size_t j = 0; j < rank; ++j) {
    global_dims[j] = gsl::narrow<uint32_t>(global_shape[j]);
    global_strides[j] = gsl::narrow<uint32_t>(global_shape.SizeFromDimension(j + 1));
  }

  const auto output_size = gsl::narrow<const uint32_t>(global_shape.Size());
  RotaryEmbeddingProgram program{interleaved_};
  const auto input_output_strides =
      input_shape.NumDimensions() == 3
          ? std::vector<uint32_t>({batch_stride, hidden_size, head_size, 1})
          : (input_shape.NumDimensions() == 4
                 ? std::vector<uint32_t>({batch_stride, head_size, sequence_length * head_size, 1})
                 : std::vector<uint32_t>({}));

  program
      .CacheHint(interleaved_)
      .AddInputs({{input, ProgramTensorMetadataDependency::Rank},
                  {position_ids, ProgramTensorMetadataDependency::Rank},
                  {cos_cache, ProgramTensorMetadataDependency::Rank},
                  {sin_cache, ProgramTensorMetadataDependency::Rank}})
      .AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{scale_},
                            {gsl::make_span(global_dims)},
                            {gsl::make_span(global_strides)},
                            {gsl::make_span(input_output_strides)}})
      .AddIndices(TensorShape{1, 1});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
