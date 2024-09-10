// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/where.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Where,
    kOnnxDomain,
    9, 15,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Where);

ONNX_OPERATOR_KERNEL_EX(
    Where,
    kOnnxDomain,
    16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Where);

// Compute where operator output shape based upon three way broad-casting.
Status ComputeOutputShape(const TensorShape& cond_shape,
                          const TensorShape& x_shape, const TensorShape& y_shape, TensorShape& output_shape) {
  size_t cond_rank = cond_shape.NumDimensions();
  size_t x_rank = x_shape.NumDimensions();
  size_t y_rank = y_shape.NumDimensions();
  size_t output_rank = std::max(std::max(cond_rank, x_rank), y_rank);

  std::vector<int64_t> output_dims(output_rank, 0);
  for (size_t i = 0; i < output_rank; ++i) {
    int64_t cond_dim = 1;
    if (i < cond_rank)
      cond_dim = cond_shape[cond_rank - 1 - i];

    int64_t x_dim = 1;
    if (i < x_rank)
      x_dim = x_shape[x_rank - 1 - i];

    int64_t y_dim = 1;
    if (i < y_rank)
      y_dim = y_shape[y_rank - 1 - i];

    int64_t output_dim = std::max(std::max(cond_dim, x_dim), y_dim);
    // special case to handle a dim of 0 which can be broadcast with a 1
    if (output_dim == 1)
      output_dim = std::min(std::min(cond_dim, x_dim), y_dim);

    const auto node_name = "Where";
    if (cond_dim != output_dim && cond_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": condition operand cannot broadcast on dim ", cond_rank - 1 - i,
                             " Condition Shape: ", cond_shape.ToString(), ", X Shape: ", x_shape.ToString(), ", Y Shape: ", y_shape.ToString());
    if (x_dim != output_dim && x_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": X operand cannot broadcast on dim ", x_rank - 1 - i,
                             " Condition Shape: ", cond_shape.ToString(), ", X Shape: ", x_shape.ToString(), ", Y Shape: ", y_shape.ToString());
    if (y_dim != output_dim && y_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": Y operand cannot broadcast on dim ", y_rank - 1 - i,
                             " Condition Shape: ", cond_shape.ToString(), ", X Shape: ", x_shape.ToString(), ", Y Shape: ", y_shape.ToString());
    output_dims[output_rank - 1 - i] = output_dim;
  }

  output_shape = TensorShape(output_dims);
  return Status::OK();
}

Status WhereProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto a_name{"a_data"};
  const auto b_name{"b_data"};
  const auto c_name{"c_data"};
  const auto output_name{"output_data"};
  const auto& c_input = shader.AddInput(c_name,
                                        ShaderVariable::UseUniform);
  const auto& a_input = shader.AddInput(a_name,
                                        ShaderVariable::UseUniform);
  const auto& b_input = shader.AddInput(b_name,
                                        ShaderVariable::UseUniform);
  const auto& output = shader.AddOutput(output_name,
                                        ShaderVariable::UseUniform);
  auto expression = [](const std::string& a, const std::string& b, const std::string& c) -> const auto {
    return "select(" + b + ", " + a + ", " + c + ")";
  };
  auto single_assignment =
      [expression, &output, &a_input, &b_input, &c_input](
          const std::string& rest_str, const std::string& x, const std::string& type_cast = "")
      -> const auto {
    const std::string a_expression = "a_data[index_a" + x + "][component_a" + x + "]";
    const std::string b_expression = "b_data[index_b" + x + "][component_b" + x + "]";
    const std::string c_expression = "bool(c_data[index_c" + x + "] & (0xffu << (component_c" + x + " * 8)))";

    std::ostringstream ss;
    ss.imbue(std::locale::classic());
    ss << "let output_indices" + x + " = " << output.OffsetToIndices("global_idx * 4u + " + x + "u") << ";\n";
    ss << "let offset_a" + x + " = " + a_input.BroadcastedIndicesToOffset("output_indices" + x, output) + ";\n";
    ss << "let offset_b" + x + " = " + b_input.BroadcastedIndicesToOffset("output_indices" + x, output) + ";\n";
    ss << "let offset_c" + x + " = " + c_input.BroadcastedIndicesToOffset("output_indices" + x, output) + ";\n";
    ss << "let index_a" + x + " = offset_a" + x + " / 4u;\n";
    ss << "let index_b" + x + " = offset_b" + x + " / 4u;\n";
    ss << "let index_c" + x + " = offset_c" + x + " / 4u;\n";
    ss << "let component_a" + x + " = offset_a" + x + " % 4u;\n";
    ss << "let component_b" + x + " = offset_b" + x + " % 4u;\n";
    ss << "let component_c" + x + " = offset_c" + x + " % 4u;\n";
    ss << rest_str + "[" + x + "] = " + type_cast + "(" + expression(a_expression, b_expression, c_expression) + ");\n";
    return ss.str();
  };

  std::string assignment;
  if (!is_broadcast_) {
    assignment = output.SetByOffset(
        "global_idx",
        expression(a_input.GetByOffset("global_idx"), b_input.GetByOffset("global_idx"), c_input.GetByOffset("global_idx")));

  } else {
    if (Outputs()[0].tensor->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
      assignment =
          "var data = vec4<u32>(0); \n" +
          single_assignment("data", "0", "u32") +
          single_assignment("data", "1", "u32") +
          single_assignment("data", "2", "u32") +
          single_assignment("data", "3", "u32") +
          "output_data[global_idx] = dot(vec4<u32>(0x1, 0x100, 0x10000, 0x1000000), vec4<u32>(data));\n";
    } else {
      assignment =
          single_assignment("output_data[global_idx]", "0") +
          single_assignment("output_data[global_idx]", "1") +
          single_assignment("output_data[global_idx]", "2") +
          single_assignment("output_data[global_idx]", "3");
    }
  }
  shader.SetMainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size"),
                             assignment);
  return Status::OK();
}

Status Where::ComputeInternal(ComputeContext& context) const {
  const auto* cond_tensor = context.Input(0);
  const auto* x_tensor = context.Input(1);
  const auto* y_tensor = context.Input(2);
  const auto& cond_shape = cond_tensor->Shape();
  const auto& x_shape = x_tensor->Shape();
  const auto& y_shape = y_tensor->Shape();

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeOutputShape(cond_shape, x_shape, y_shape, output_shape));
  auto* output_tensor = context.Output(0, output_shape);
  const auto component = 4;
  uint32_t vec_size = gsl::narrow_cast<uint32_t>((output_shape.Size() + 3) / component);
  const auto is_broadcast = !(x_shape == y_shape &&
                              y_shape == cond_shape);
  WhereProgram program{is_broadcast};
  program
      .CacheHint(is_broadcast)
      .AddInputs({{cond_tensor, ProgramTensorMetadataDependency::Rank, component},
                  {x_tensor, ProgramTensorMetadataDependency::Rank, component},
                  {y_tensor, ProgramTensorMetadataDependency::Rank, component}})
      .AddOutputs({{output_tensor,
                    ProgramTensorMetadataDependency::Rank |
                        ProgramTensorMetadataDependency::Type,
                    component}})
      .SetDispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
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

WEBGPU_TRANSPOSE_VERSIONED_KERNEL(Where, 9, 15, Where, WebGpuSupportedFloatTypes())
WEBGPU_TRANSPOSE_KERNEL(Where, 16, Where, WebGpuSupportedFloatTypes())

}  // namespace webgpu
}  // namespace onnxruntime
