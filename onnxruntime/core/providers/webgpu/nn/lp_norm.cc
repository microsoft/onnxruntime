// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/nn/lp_norm.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace webgpu {

Status LpNormProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("y", ShaderUsage::UseUniform);

  shader.AdditionalImplementation()
      << "var<workgroup> norm_shared : array<f32, workgroup_size_x>;\n";

  shader.MainFunctionBody()
      << "let ix = local_idx;\n"
      << "let norm_group = workgroup_idx;\n"
      << "if (norm_group >= uniforms.norm_count) { return; }\n"

      // Compute base offset for this norm group.
      // Elements in the norm group are at: base + j * stride_factor, j = 0..norm_size-1
      << "let base = (norm_group / uniforms.stride_factor) * uniforms.stride_factor * uniforms.norm_size"
      << " + (norm_group % uniforms.stride_factor);\n"

      // Distribute elements across threads
      << "var elements_per_thread = uniforms.norm_size / workgroup_size_x;\n"
      << "let remainder = uniforms.norm_size % workgroup_size_x;\n"
      << "var start: u32 = 0u;\n"
      << "if (ix < remainder) {\n"
      << "  elements_per_thread = elements_per_thread + 1u;\n"
      << "  start = ix * elements_per_thread;\n"
      << "} else {\n"
      << "  start = ix * elements_per_thread + remainder;\n"
      << "}\n"

      // Phase 1: Accumulate norm contribution
      << "var local_sum: f32 = 0.0;\n"
      << "for (var j: u32 = 0u; j < elements_per_thread; j++) {\n"
      << "  let val = f32(x[base + (start + j) * uniforms.stride_factor]);\n";

  if (p_ == 1) {
    shader.MainFunctionBody()
        << "  local_sum += abs(val);\n";
  } else {
    shader.MainFunctionBody()
        << "  local_sum += val * val;\n";
  }

  shader.MainFunctionBody()
      << "}\n"
      << "norm_shared[ix] = local_sum;\n"
      << "workgroupBarrier();\n"

      // Phase 2: Parallel reduction
      << "var reduce_size : u32 = workgroup_size_x;\n"
      << "for (var curr_size = reduce_size >> 1; curr_size > 0; curr_size = reduce_size >> 1) {\n"
      << "  reduce_size = curr_size + (reduce_size & 1);\n"
      << "  if (ix < curr_size) {\n"
      << "    norm_shared[ix] += norm_shared[ix + reduce_size];\n"
      << "  }\n"
      << "  workgroupBarrier();\n"
      << "}\n";

  // Phase 3: Compute norm value
  if (p_ == 1) {
    shader.MainFunctionBody()
        << "let norm_val = norm_shared[0];\n";
  } else {
    shader.MainFunctionBody()
        << "let norm_val = sqrt(norm_shared[0]);\n";
  }

  // Phase 4: Normalize
  shader.MainFunctionBody()
      << "for (var j: u32 = 0u; j < elements_per_thread; j++) {\n"
      << "  let offset = base + (start + j) * uniforms.stride_factor;\n"
      << "  if (norm_val != 0.0) {\n"
      << "    y[offset] = x_element_t(f32(x[offset]) / norm_val);\n"
      << "  } else {\n"
      << "    y[offset] = x_element_t(0.0);\n"
      << "  }\n"
      << "}\n";

  return Status::OK();
}

Status LpNorm::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input(0);
  const auto x_shape = x->Shape();

  if (x_shape.Size() == 0) {
    context.Output(0, x_shape);
    return Status::OK();
  }

  const auto rank = static_cast<int64_t>(x_shape.NumDimensions());
  const auto canonical_axis = HandleNegativeAxis(axis_, rank);

  const uint32_t m = onnxruntime::narrow<uint32_t>(x_shape.GetDims()[onnxruntime::narrow<size_t>(canonical_axis)]);
  const uint32_t n = onnxruntime::narrow<uint32_t>(x_shape.Size() / m);
  const uint32_t sf = (canonical_axis + 1 < rank)
                          ? onnxruntime::narrow<uint32_t>(x_shape.SizeFromDimension(onnxruntime::narrow<size_t>(canonical_axis) + 1))
                          : 1u;

  auto* y = context.Output(0, x_shape);

  TensorShape override_shape{x_shape.Size()};

  LpNormProgram program{p_};
  program.CacheHint(p_)
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, override_shape, 1}})
      .AddOutputs({{y, ProgramTensorMetadataDependency::None, override_shape, 1}})
      .AddUniformVariables({{n}})
      .AddUniformVariables({{m}})
      .AddUniformVariables({{sf}});

  program.SetDispatchGroupSize(n);

  return context.RunProgram(program);
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(LpNormalization, kOnnxDomain, 1, 21, kWebGpuExecutionProvider,
                                  (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
                                  LpNorm);

ONNX_OPERATOR_KERNEL_EX(LpNormalization, kOnnxDomain, 22, kWebGpuExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
                        LpNorm);

}  // namespace webgpu
}  // namespace onnxruntime
