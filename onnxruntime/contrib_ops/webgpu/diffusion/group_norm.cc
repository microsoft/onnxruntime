// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/diffusion/group_norm.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using onnxruntime::webgpu::ProgramTensorMetadataDependency;
using onnxruntime::webgpu::ShaderUsage;
using onnxruntime::webgpu::ShaderVariableHelper;
using onnxruntime::webgpu::SumVector;
using onnxruntime::webgpu::WebGpuSupportedFloatTypes;
using onnxruntime::webgpu::WORKGROUP_SIZE;

Status GroupNormStatsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform);
  const ShaderVariableHelper* skip = has_skip_ ? &shader.AddInput("skip", ShaderUsage::UseUniform) : nullptr;
  const ShaderVariableHelper* bias = has_bias_ ? &shader.AddInput("bias", ShaderUsage::UseUniform) : nullptr;
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  shader.AdditionalImplementation() << "alias f32_val_t = " << (components_ == 4 ? "vec4<f32>" : (components_ == 2 ? "vec2<f32>" : "f32")) << ";\n"
                                    << "var<workgroup> workgroup_shared_sum : array<f32_val_t, " << workgroup_size_ << ">;\n"
                                    << "var<workgroup> workgroup_shared_squared_sum : array<f32_val_t, " << workgroup_size_ << ">;\n"
                                    << "const workgroup_size = " << workgroup_size_ << "u;\n";

  shader.MainFunctionBody() << "  let n = workgroup_idx / uniforms.groups;\n"
                            << "  let g = workgroup_idx % uniforms.groups;\n"
                            << "  let count = uniforms.hw * uniforms.cg_comp;\n"
                            << "  var sum = f32_val_t(0);\n"
                            << "  var squared_sum = f32_val_t(0);\n"
                            << "  for (var i = local_idx; i < count; i += workgroup_size) {\n"
                            << "    let hw = i / uniforms.cg_comp;\n"
                            << "    let k = i % uniforms.cg_comp;\n"
                            << "    let c_idx = g * uniforms.cg_comp + k;\n"
                            << "    let offset = (n * uniforms.hw + hw) * uniforms.c_comp + c_idx;\n"
                            << "    var value = f32_val_t(" << x.GetByOffset("offset") << ");\n";
  if (has_skip_) {
    shader.MainFunctionBody() << "    value += f32_val_t("
                              << skip->GetByOffset(skip_broadcast_ ? "n * uniforms.c_comp + c_idx" : "offset") << ");\n";
  }
  if (has_bias_) {
    shader.MainFunctionBody() << "    value += f32_val_t(" << bias->GetByOffset("c_idx") << ");\n";
  }
  shader.MainFunctionBody() << "    sum += value;\n"
                            << "    squared_sum += value * value;\n"
                            << "  }\n"
                            << "  workgroup_shared_sum[local_idx] = sum;\n"
                            << "  workgroup_shared_squared_sum[local_idx] = squared_sum;\n"
                            << "  workgroupBarrier();\n"
                            << "  for (var curr_size = workgroup_size >> 1; curr_size > 0; curr_size = curr_size >> 1) {\n"
                            << "    if (local_idx < curr_size) {\n"
                            << "      workgroup_shared_sum[local_idx] = workgroup_shared_sum[local_idx] + workgroup_shared_sum[local_idx + curr_size];\n"
                            << "      workgroup_shared_squared_sum[local_idx] = workgroup_shared_squared_sum[local_idx] + workgroup_shared_squared_sum[local_idx + curr_size];\n"
                            << "    }\n"
                            << "    workgroupBarrier();\n"
                            << "  }\n"
                            << "  if (local_idx == 0) {\n"
                            << "    let element_count = f32(count * " << components_ << "u);\n"
                            << "    let mean = " << SumVector("workgroup_shared_sum[0]", components_) << " / element_count;\n"
                            << "    let squared_mean = " << SumVector("workgroup_shared_squared_sum[0]", components_) << " / element_count;\n"
                            << "    let inv_std_dev = inverseSqrt(squared_mean - mean * mean + uniforms.epsilon);\n"
                            << "    " << output.SetByOffset("workgroup_idx", "output_value_t(mean, inv_std_dev)") << ";\n"
                            << "  }\n";
  return Status::OK();
}

Status GroupNormApplyProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform);
  const ShaderVariableHelper* skip = has_skip_ ? &shader.AddInput("skip", ShaderUsage::UseUniform) : nullptr;
  const ShaderVariableHelper* bias = has_bias_ ? &shader.AddInput("bias", ShaderUsage::UseUniform) : nullptr;
  const auto& stats = shader.AddInput("stats", ShaderUsage::UseUniform);
  const auto& gamma = shader.AddInput("gamma", ShaderUsage::UseUniform);
  const auto& beta = shader.AddInput("beta", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const ShaderVariableHelper* sum_output = has_sum_output_ ? &shader.AddOutput("sum_output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias) : nullptr;

  shader.AdditionalImplementation() << "alias f32_val_t = " << (components_ == 4 ? "vec4<f32>" : (components_ == 2 ? "vec2<f32>" : "f32")) << ";\n";

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let hwc = uniforms.hw * uniforms.c_comp;\n"
                            << "  let n = global_idx / hwc;\n"
                            << "  let c_idx = global_idx % uniforms.c_comp;\n"
                            << "  let g = c_idx / uniforms.cg_comp;\n"
                            << "  let mean_inv_std = " << stats.GetByOffset("n * uniforms.groups + g") << ";\n"
                            << "  let gamma_v = f32_val_t(" << gamma.GetByOffset("c_idx") << ");\n"
                            << "  let beta_v = f32_val_t(" << beta.GetByOffset("c_idx") << ");\n"
                            << "  var value = f32_val_t(" << x.GetByOffset("global_idx") << ");\n";
  if (has_skip_) {
    shader.MainFunctionBody() << "  value += f32_val_t("
                              << skip->GetByOffset(skip_broadcast_ ? "n * uniforms.c_comp + c_idx" : "global_idx") << ");\n";
  }
  if (has_bias_) {
    shader.MainFunctionBody() << "  value += f32_val_t(" << bias->GetByOffset("c_idx") << ");\n";
  }
  if (has_sum_output_) {
    shader.MainFunctionBody() << "  " << sum_output->SetByOffset("global_idx", "sum_output_value_t(value)") << ";\n";
  }
  shader.MainFunctionBody() << "  var result = (value - mean_inv_std.x) * mean_inv_std.y * gamma_v + beta_v;\n";
  if (use_silu_) {
    shader.MainFunctionBody() << "  result = result * (f32_val_t(1) / (f32_val_t(1) + exp(-result)));\n";
  }
  shader.MainFunctionBody() << "  " << output.SetByOffset("global_idx", "output_value_t(result)") << ";\n";
  return Status::OK();
}

Status GroupNorm::ComputeInternal(ComputeContext& context) const {
  const auto* x = context.Input<Tensor>(0);
  const auto* gamma = context.Input<Tensor>(1);
  const auto* beta = context.Input<Tensor>(2);
  const auto* skip = context.Input<Tensor>(3);  // SkipGroupNorm only
  const auto* bias = context.Input<Tensor>(4);  // SkipGroupNorm only

  ORT_RETURN_IF_NOT(channels_last_ == 1, "WebGPU GroupNorm only supports channels_last=1.");

  const auto& x_shape = x->Shape();
  const auto rank = x_shape.NumDimensions();
  ORT_RETURN_IF_NOT(rank >= 3, "GroupNorm input must have rank >= 3 (N, spatial..., C).");

  const bool has_skip = skip != nullptr;
  const bool has_bias = bias != nullptr;
  Tensor* y = context.Output(0, x_shape);
  Tensor* sum_output = has_skip ? context.Output(1, x_shape) : nullptr;
  const bool has_sum_output = sum_output != nullptr;

  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  const int64_t batch = x_shape[0];
  const int64_t channels = x_shape[rank - 1];
  const int64_t hw = x_shape.SizeFromDimension(1) / channels;
  const int64_t groups = groups_;
  ORT_RETURN_IF_NOT(channels % groups == 0, "Number of channels must be divisible by groups.");
  const int64_t channels_per_group = channels / groups;

  ORT_RETURN_IF_NOT(gamma->Shape().Size() == channels && beta->Shape().Size() == channels,
                    "gamma and beta must have size equal to number of channels.");
  // skip is either the same shape as X, or per-(batch, channel) broadcast: (N, C) or (N, 1, 1, C).
  bool skip_broadcast = false;
  if (has_skip) {
    const auto& skip_shape = skip->Shape();
    if (skip_shape == x_shape) {
      skip_broadcast = false;
    } else {
      const auto skip_rank = skip_shape.NumDimensions();
      // Check the rank first so the short-circuit protects the indexing below.
      const bool valid_broadcast =
          (skip_rank == 2 || skip_rank == rank) &&
          skip_shape[0] == batch && skip_shape[skip_rank - 1] == channels &&
          skip_shape.Size() == batch * channels;
      ORT_RETURN_IF_NOT(valid_broadcast,
                        "SkipGroupNorm skip must have the same shape as X, or be broadcastable as (N, C) or (N, 1, ..., 1, C).");
      skip_broadcast = true;
    }
  }
  ORT_RETURN_IF_NOT(!has_bias || bias->Shape().Size() == channels,
                    "SkipGroupNorm bias must be a 1D tensor with size equal to number of channels.");

  const int components = channels_per_group % 4 == 0 ? 4 : (channels_per_group % 2 == 0 ? 2 : 1);
  const int64_t c_comp = channels / components;
  const int64_t cg_comp = channels_per_group / components;

  // Pass 1: per-(batch, group) mean / inv_std, f32, shape [N * G, 2] accessed with 2 components.
  TensorShape stats_shape{batch * groups, 2};
  Tensor stats = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), stats_shape);
  const uint32_t stats_workgroup_size = 256;
  const TensorShape x_flat_shape{batch * hw * c_comp};
  const TensorShape skip_flat_shape{skip_broadcast ? batch * c_comp : batch * hw * c_comp};
  const TensorShape channel_flat_shape{c_comp};

  GroupNormStatsProgram stats_program{components, stats_workgroup_size, has_skip, skip_broadcast, has_bias};
  stats_program.CacheHint(components, stats_workgroup_size, has_skip, skip_broadcast, has_bias)
      .AddInput({x, ProgramTensorMetadataDependency::Type, x_flat_shape, components})
      .AddOutput({&stats, ProgramTensorMetadataDependency::None, TensorShape{batch * groups, 1}, 2})
      .SetDispatchGroupSize(static_cast<uint32_t>(batch * groups))
      .SetWorkgroupSize(stats_workgroup_size)
      .AddUniformVariables({{static_cast<uint32_t>(hw)},
                            {static_cast<uint32_t>(c_comp)},
                            {static_cast<uint32_t>(cg_comp)},
                            {static_cast<uint32_t>(groups)},
                            {epsilon_}});
  if (has_skip) {
    stats_program.AddInput({skip, ProgramTensorMetadataDependency::Type, skip_flat_shape, components});
  }
  if (has_bias) {
    stats_program.AddInput({bias, ProgramTensorMetadataDependency::Type, channel_flat_shape, components});
  }
  ORT_RETURN_IF_ERROR(context.RunProgram(stats_program));

  // Pass 2: normalize + affine + optional SiLU, elementwise; optionally writes x + skip + bias to S.
  const bool use_silu = activation_ == 1;
  const int64_t output_size = batch * hw * c_comp;
  GroupNormApplyProgram apply_program{components, use_silu, has_skip, skip_broadcast, has_bias, has_sum_output};
  apply_program.CacheHint(components, use_silu, has_skip, skip_broadcast, has_bias, has_sum_output)
      .AddInputs({{x, ProgramTensorMetadataDependency::Type, x_flat_shape, components}})
      .SetDispatchGroupSize(static_cast<uint32_t>((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE))
      .AddUniformVariables({{static_cast<uint32_t>(output_size)},
                            {static_cast<uint32_t>(hw)},
                            {static_cast<uint32_t>(c_comp)},
                            {static_cast<uint32_t>(cg_comp)},
                            {static_cast<uint32_t>(groups)}});
  if (has_skip) {
    apply_program.AddInput({skip, ProgramTensorMetadataDependency::Type, skip_flat_shape, components});
  }
  if (has_bias) {
    apply_program.AddInput({bias, ProgramTensorMetadataDependency::Type, channel_flat_shape, components});
  }
  apply_program.AddInputs({{&stats, ProgramTensorMetadataDependency::None, TensorShape{batch * groups, 1}, 2},
                           {gamma, ProgramTensorMetadataDependency::Type, TensorShape{c_comp}, components},
                           {beta, ProgramTensorMetadataDependency::Type, TensorShape{c_comp}, components}});
  apply_program.AddOutput({y, ProgramTensorMetadataDependency::None, x_flat_shape, components});
  if (has_sum_output) {
    apply_program.AddOutput({sum_output, ProgramTensorMetadataDependency::None, x_flat_shape, components});
  }
  return context.RunProgram(apply_program);
}

ONNX_OPERATOR_KERNEL_EX(
    GroupNorm,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", WebGpuSupportedFloatTypes()),
    GroupNorm);

ONNX_OPERATOR_KERNEL_EX(
    SkipGroupNorm,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("M", WebGpuSupportedFloatTypes()),
    GroupNorm);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
