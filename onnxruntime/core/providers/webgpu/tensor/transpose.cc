// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    1, 12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Transpose);

ONNX_OPERATOR_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Transpose);

ONNX_OPERATOR_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    17,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Transpose);

const std::string permFunctionBody(const std::string& input_name, const std::string& output_name, const gsl::span<const size_t>& perm) {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

  ss << "fn perm(i: " << output_name << "_indices_t"
     << ")->" << input_name << "_indices_t "
     << "{\n  var a: " << input_name << "_indices_t;\n";
  for (auto i = 0; i < perm.size(); ++i) {
    ss << "  a[" << perm[i] << "] = i[" << i << "];\n";
  }
  ss << "  return a;\n}\n";
  return ss.str();
}

Status TransposeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto input_name{"x"};
  const auto output_name{"y"};
  const auto& input = shader.AddInput(input_name,
                                      ToProgramVariableDataType(Inputs()[0].tensor->GetElementType()),
                                      ShaderVariable::UseUniform | ShaderVariable::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput(output_name,
                                        ToProgramVariableDataType(Outputs()[0].tensor->GetElementType()),
                                        ShaderVariable::UseUniform | ShaderVariable::UseIndicesTypeAlias);
  shader.AppendImplementation(permFunctionBody(input_name, output_name, this->perm_));
  shader.MainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size"),
                          "  let indices = ", output.OffsetToIndices("global_idx"),";\n", "  let x_indices = perm(indices); \n",
                          output.SetByOffset("global_idx", input.GetByIndices("x_indices")));
  return Status::OK();
}

Status Transpose::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  int32_t rank = gsl::narrow_cast<int32_t>(input_shape.NumDimensions());

  TensorShapeVector output_dims(rank);
  InlinedVector<size_t> default_perm(rank);
  const InlinedVector<size_t>* p_perm = nullptr;
  const auto& status = ComputeOutputShape(*input_tensor, output_dims, default_perm, p_perm);
  if (!status.IsOK())
    return status;
  TensorShape output_shape(output_dims);
  auto* output_tensor = context.Output(0, output_shape);

  SafeInt<uint32_t> vec_size = input_tensor->Shape().Size();
  TransposeProgram program{"Transpose", *p_perm};
  program
      .Inputs({{input_tensor, ProgramTensorMetadataDependency::Rank}})
      .Outputs({output_tensor})
      .DispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .UniformVariables({
          {static_cast<uint32_t>(vec_size)},
      });
  return context.RunProgram(program);
}

#define WEBGPU_TRANSPOSE_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_KERNEL_EX(                                            \
      OP_TYPE, kOnnxDomain, VERSION, kWebGpuExecutionProvider,        \
      KernelDefBuilder().TypeConstraint("T", TYPE),                   \
      KERNEL_CLASS);

#define WEBGPU_TRANSPOSE_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                             \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kWebGpuExecutionProvider,                  \
      KernelDefBuilder().TypeConstraint("T", TYPE),                                              \
      KERNEL_CLASS);

WEBGPU_TRANSPOSE_VERSIONED_KERNEL(Transpose, 1, 12, Transpose, WebGpuSupportedFloatTypes())
WEBGPU_TRANSPOSE_KERNEL(Transpose, 13, Transpose, WebGpuSupportedFloatTypes())
WEBGPU_TRANSPOSE_KERNEL(Transpose, 17, Transpose, WebGpuSupportedFloatTypes())

}  // namespace webgpu
}  // namespace onnxruntime
