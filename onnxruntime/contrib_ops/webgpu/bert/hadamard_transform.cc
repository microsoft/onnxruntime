// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/hadamard_transform.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status HadamardTransformProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& input = sh.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& output = sh.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  return WGSL_TEMPLATE_APPLY(sh, "bert/hadamard_transform.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(components, components_),
                             WGSL_TEMPLATE_PARAMETER(head_size_log2, head_size_log2_),
                             WGSL_TEMPLATE_VARIABLE(input, input),
                             WGSL_TEMPLATE_VARIABLE(output, output));
}

Status ApplyHadamardTransform(onnxruntime::webgpu::ComputeContext& context,
                              const Tensor* input,
                              Tensor* output,
                              int batch_size,
                              int seq_length,
                              int num_heads,
                              int head_size) {
  // head_size must be a power of 2
  ORT_ENFORCE((head_size & (head_size - 1)) == 0, "head_size must be a power of 2 for Hadamard transform, got ", head_size);
  ORT_ENFORCE(head_size >= 4, "head_size must be at least 4 for vectorized Hadamard transform, got ", head_size);

  int head_size_log2 = 0;
  for (int tmp = head_size; tmp > 1; tmp >>= 1) {
    head_size_log2++;
  }

  const int components = head_size % 4 == 0 ? 4 : (head_size % 2 == 0 ? 2 : 1);
  const uint32_t num_slices = static_cast<uint32_t>(batch_size * seq_length * num_heads);

  // Workgroup size: use up to 64 threads. Each thread handles multiple butterfly pairs.
  const uint32_t workgroup_size = std::min(static_cast<uint32_t>(head_size / 2), 64u);

  HadamardTransformProgram program(head_size_log2, components);
  program.AddInput({input, ProgramTensorMetadataDependency::TypeAndRank, components});
  program.AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, components});

  program.SetDispatchGroupSize(num_slices)
      .SetWorkgroupSize(workgroup_size)
      .CacheHint(head_size_log2, components)
      .AddUniformVariables({{static_cast<uint32_t>(batch_size)},
                            {static_cast<uint32_t>(seq_length)},
                            {static_cast<uint32_t>(num_heads)}});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
