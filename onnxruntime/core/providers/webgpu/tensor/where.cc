// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/where.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

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

    int64_t output_dim = std::max({cond_dim, x_dim, y_dim});
    // special case to handle a dim of 0 which can be broadcast with a 1
    if (output_dim == 1)
      output_dim = std::min({cond_dim, x_dim, y_dim});

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
  const auto& c_input = shader.AddInput("c_data", ShaderUsage::UseUniform);
  const auto& a_input = shader.AddInput("a_data", ShaderUsage::UseUniform);
  const auto& b_input = shader.AddInput("b_data", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output_data", ShaderUsage::UseUniform);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size");

  const auto expression = [](std::string_view a, std::string_view b, std::string_view c) -> auto {
    return absl::StrCat("select(", b, ", ", a, ", ", c, ")");
  };

  if (!is_broadcast_) {
    shader.MainFunctionBody() << output.SetByOffset(
        "global_idx",
        expression(a_input.GetByOffset("global_idx"), b_input.GetByOffset("global_idx"), c_input.GetByOffset("global_idx")));

  } else {
    const auto& c_indices = shader.AddIndices("c_indices");
    const auto& a_indices = shader.AddIndices("a_indices");
    const auto& b_indices = shader.AddIndices("b_indices");
    const auto& output_indices = shader.AddIndices("output_indices");

    const auto single_assignment =
        [&expression, &shader, &output_indices, &a_indices, &b_indices, &c_indices](
            std::string_view rest_str, const std::string& x, std::string_view type_cast = "")
        -> void {
      const std::string a_expression = "a_data[index_a" + x + "][component_a" + x + "]";
      const std::string b_expression = "b_data[index_b" + x + "][component_b" + x + "]";
      const std::string c_expression = "bool(c_data[index_c" + x + "] & (0xffu << (component_c" + x + " * 8)))";

      shader.MainFunctionBody() << "let output_indices" << x << " = " << output_indices.OffsetToIndices("global_idx * 4 + " + x) << ";\n"
                                << "let offset_a" << x << " = " << a_indices.BroadcastedIndicesToOffset("output_indices" + x, output_indices) << ";\n"
                                << "let offset_b" << x << " = " << b_indices.BroadcastedIndicesToOffset("output_indices" + x, output_indices) << ";\n"
                                << "let offset_c" << x << " = " << c_indices.BroadcastedIndicesToOffset("output_indices" + x, output_indices) << ";\n"
                                << "let index_a" << x << " = offset_a" << x << " / 4;\n"
                                << "let index_b" << x << " = offset_b" << x << " / 4;\n"
                                << "let index_c" << x << " = offset_c" << x << " / 4;\n"
                                << "let component_a" << x << " = offset_a" << x << " % 4;\n"
                                << "let component_b" << x << " = offset_b" << x << " % 4;\n"
                                << "let component_c" << x << " = offset_c" << x << " % 4;\n"
                                << rest_str << "[" << x << "] = " << type_cast << "(" << expression(a_expression, b_expression, c_expression) << ");\n";
    };

    if (Outputs()[0].tensor->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
      shader.MainFunctionBody() << "var data = vec4<u32>(0);\n";
      single_assignment("data", "0", "u32");
      single_assignment("data", "1", "u32");
      single_assignment("data", "2", "u32");
      single_assignment("data", "3", "u32");
      shader.MainFunctionBody() << "output_data[global_idx] = dot(vec4<u32>(0x1, 0x100, 0x10000, 0x1000000), vec4<u32>(data));\n";
    } else {
      single_assignment("output_data[global_idx]", "0");
      single_assignment("output_data[global_idx]", "1");
      single_assignment("output_data[global_idx]", "2");
      single_assignment("output_data[global_idx]", "3");
    }
  }

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
  constexpr int component = 4;
  uint32_t vec_size = gsl::narrow_cast<uint32_t>((output_shape.Size() + 3) / component);
  const auto is_broadcast = !(x_shape == y_shape &&
                              y_shape == cond_shape);
  WhereProgram program{is_broadcast};
  program
      .CacheHint(is_broadcast)
      .SetDispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddInputs({{cond_tensor, ProgramTensorMetadataDependency::Rank, {(cond_shape.Size() + 3) / 4}, 4},
                  {x_tensor, ProgramTensorMetadataDependency::Rank, {(x_shape.Size() + 3) / 4}, 4},
                  {y_tensor, ProgramTensorMetadataDependency::Rank, {(y_shape.Size() + 3) / 4}, 4}})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Type, {vec_size}, 4})
      .AddUniformVariables({
          {static_cast<uint32_t>(vec_size)},
      });
  if (is_broadcast) {
    program
        .AddIndices(cond_shape)
        .AddIndices(x_shape)
        .AddIndices(y_shape)
        .AddIndices(output_tensor->Shape());
  }
  return context.RunProgram(program);
}

namespace {
const std::vector<MLDataType>& WhereOpTypeConstraints() {
  // currently support boolean, integer and float types that explicitly allowed in WGSL:
  // https://gpuweb.github.io/gpuweb/wgsl/#plain-types-section
  //
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<MLFloat16>(),
      DataTypeImpl::GetTensorType<float>(),
      DataTypeImpl::GetTensorType<int32_t>(),
      DataTypeImpl::GetTensorType<uint32_t>(),
      DataTypeImpl::GetTensorType<bool>()};
  return types;
}
}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Where,
    kOnnxDomain,
    9, 15,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WhereOpTypeConstraints()),
    Where);

ONNX_OPERATOR_KERNEL_EX(
    Where,
    kOnnxDomain,
    16,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WhereOpTypeConstraints()),
    Where);

}  // namespace webgpu
}  // namespace onnxruntime
