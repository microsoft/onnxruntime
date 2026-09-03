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
                             WGSL_TEMPLATE_PARAMETER(hadamard_size_log2, slice_size_log2_),
                             WGSL_TEMPLATE_VARIABLE(input, input),
                             WGSL_TEMPLATE_VARIABLE(output, output));
}

Status ApplyHadamardTransform(onnxruntime::webgpu::ComputeContext& context,
                              const Tensor* input,
                              Tensor* output,
                              int explicit_slice_size) {
  const auto& shape = input->Shape();
  ORT_ENFORCE(shape.NumDimensions() >= 1, "Input tensor must have at least 1 dimension.");

  // Use explicit slice size if provided, otherwise derive from last dimension.
  const int slice_size = explicit_slice_size > 0 ? explicit_slice_size : static_cast<int>(shape[shape.NumDimensions() - 1]);
  ORT_ENFORCE((slice_size & (slice_size - 1)) == 0, "Last dimension must be a power of 2 for Hadamard transform, got ", slice_size);
  ORT_ENFORCE(slice_size >= 4, "Last dimension must be at least 4 for vectorized Hadamard transform, got ", slice_size);

  const int slice_size_log2 = Log2OfPowerOfTwo(slice_size);

  const int components = slice_size % 4 == 0 ? 4 : (slice_size % 2 == 0 ? 2 : 1);
  ORT_ENFORCE(shape.Size() % slice_size == 0, "Total tensor size must be divisible by slice_size, got ", shape.Size(), " % ", slice_size, " != 0");
  const uint32_t num_slices = static_cast<uint32_t>(shape.Size() / slice_size);

  // Workgroup size: use up to 64 threads. Each thread handles multiple butterfly pairs.
  const uint32_t workgroup_size = std::min(static_cast<uint32_t>(slice_size / 2), 64u);

  HadamardTransformProgram program(slice_size_log2, components);
  program.AddInput({input, ProgramTensorMetadataDependency::TypeAndRank, components});
  program.AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, components});

  program.SetDispatchGroupSize(num_slices)
      .SetWorkgroupSize(workgroup_size)
      .CacheHint(slice_size_log2, components)
      .AddUniformVariables({{num_slices}});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
