// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/ngram_hash_mapping.h"

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    NgramHashMapping,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()),
    NgramHashMapping);

Status NgramHashMappingProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input_ids = shader.AddInput("input_ids", ShaderUsage::UseUniform);
  const auto& multipliers = shader.AddInput("multipliers", ShaderUsage::UseUniform);
  const auto& vocab_sizes = shader.AddInput("vocab_sizes", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let num_heads = (uniforms.max_ngram_size - 1u) * uniforms.n_head_per_ngram;\n"
      << "  let t = global_idx % uniforms.sequence_length;\n"
      << "  let b = global_idx / uniforms.sequence_length;\n"
      << "  let input_base = b * uniforms.sequence_length;\n"
      << "  let output_base = global_idx * num_heads;\n"
      << "  for (var n = 2u; n <= uniforms.max_ngram_size; n++) {\n"
      << "    var mix = 0i;\n"
      << "    for (var k = 0u; k < n; k++) {\n"
      << "      var token = uniforms.pad_id;\n"
      << "      if (t >= k) {\n"
      << "        token = " << input_ids.GetByOffset("input_base + t - k") << ";\n"
      << "      }\n"
      << "      let product = token * " << multipliers.GetByOffset("k") << ";\n"
      << "      if (k == 0u) { mix = product; } else { mix = mix ^ product; }\n"
      << "    }\n"
      << "    let ngram_offset = (n - 2u) * uniforms.n_head_per_ngram;\n"
      << "    for (var h = 0u; h < uniforms.n_head_per_ngram; h++) {\n"
      << "      let out_h = ngram_offset + h;\n"
      << "      let mod_value = " << vocab_sizes.GetByOffset("out_h") << ";\n"
      << "      var result = 0i;\n"
      << "      if (mod_value > 0i) {\n"
      << "        result = mix % mod_value;\n"
      << "        if (result < 0i) { result += mod_value; }\n"
      << "      }\n"
      << "      " << output.SetByOffset("output_base + out_h", "result") << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

NgramHashMapping::NgramHashMapping(const OpKernelInfo& info) : WebGpuKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("max_ngram_size", &max_ngram_size_).IsOK(),
              "max_ngram_size attribute is required");
  ORT_ENFORCE(info.GetAttr<int64_t>("n_head_per_ngram", &n_head_per_ngram_).IsOK(),
              "n_head_per_ngram attribute is required");
  ORT_ENFORCE(info.GetAttr<int64_t>("pad_id", &pad_id_).IsOK(), "pad_id attribute is required");
  ORT_ENFORCE(max_ngram_size_ >= 2, "max_ngram_size must be at least 2");
  ORT_ENFORCE(n_head_per_ngram_ >= 1, "n_head_per_ngram must be positive");
  ORT_ENFORCE(pad_id_ >= std::numeric_limits<int32_t>::min() && pad_id_ <= std::numeric_limits<int32_t>::max(),
              "WebGPU NgramHashMapping only supports int32 ids");
}

Status NgramHashMapping::ComputeInternal(ComputeContext& context) const {
  const auto* input_ids = context.Input(0);
  const auto* multipliers = context.Input(1);
  const auto* vocab_sizes = context.Input(2);
  const auto& input_shape = input_ids->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2, "input_ids must have rank 2");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 && multipliers->Shape()[0] >= max_ngram_size_,
                    "multipliers must have shape (max_ngram_size)");
  const int64_t num_heads = (max_ngram_size_ - 1) * n_head_per_ngram_;
  ORT_RETURN_IF_NOT(vocab_sizes->Shape() == TensorShape({num_heads}),
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");
  auto* output = context.Output(0, TensorShape({input_shape[0], input_shape[1], num_heads}));
  const int64_t total = input_shape.Size();
  if (total == 0) {
    return Status::OK();
  }

  NgramHashMappingProgram program;
  program.AddInputs({{input_ids, ProgramTensorMetadataDependency::None},
                     {multipliers, ProgramTensorMetadataDependency::None},
                     {vocab_sizes, ProgramTensorMetadataDependency::None}})
      .AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total)},
                            {onnxruntime::narrow<uint32_t>(input_shape[1])},
                            {onnxruntime::narrow<uint32_t>(max_ngram_size_)},
                            {onnxruntime::narrow<uint32_t>(n_head_per_ngram_)},
                            {onnxruntime::narrow<int32_t>(pad_id_)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
