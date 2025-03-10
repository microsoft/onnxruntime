// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/math/unary_elementwise_ops.cc"  // contains Gelu definition
// #include "contrib_ops/webgpu/bert/gelu.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

ONNX_OPERATOR_KERNEL_EX(
    Gelu,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    onnxruntime::webgpu::Gelu);

// Status GeluProgram::GenerateShaderCode(ShaderHelper& shader) const {
//   const auto& x = shader.AddInput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
//   const auto& y = shader.AddOutput("y", ShaderUsage::UseUniform);

//   shader.AdditionalImplementation() << ErfImpl;
//   shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size")
//                             << "  let a = " << x.GetByOffset("global_idx") << ";\n"
//                             << y.SetByOffset("global_idx", onnxruntime::webgpu::GeluExpr);

//   return Status::OK();
// }

// Status Gelu::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
//   const auto* input = context.Input(0);
//   auto* output = context.Output(0, input->Shape());

//   uint32_t data_size = gsl::narrow<uint32_t>(output->Shape().Size());
//   if (data_size == 0) {
//     return Status::OK();
//   }

//   const auto vec_size = (data_size + 3) / 4;

//   GeluProgram program{};
//   program.AddInput({input, ProgramTensorMetadataDependency::Type, {vec_size}, 4})
//       .AddOutput({output, ProgramTensorMetadataDependency::None, {vec_size}, 4})
//       .SetDispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
//       .AddUniformVariable({vec_size});
//   return context.RunProgram(program);
// }

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime