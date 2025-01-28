// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/webgpu/math/binary_elementwise_ops.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {
Status BinaryElementwiseProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& b = shader.AddInput("input_b", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  const auto& c = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size");

  // check whether can use element-wise mode.
  // If either A or B is scalar, or A and B have the same shape, element-wise mode can be used.
  // In element-wise mode, no indices calculation is needed.
  if (is_lhs_scalar_ || is_rhs_scalar_ || !is_broadcast_) {
    // get A data
    if (is_lhs_scalar_) {
      shader.MainFunctionBody() << "let a = input_a_value_t(" << a.GetByOffset("0") << ".x);\n";
    } else {
      shader.MainFunctionBody() << "let a = " << a.GetByOffset("global_idx") << ";\n";
    }

    // get B data
    if (is_rhs_scalar_) {
      shader.MainFunctionBody() << "let b = input_b_value_t(" << b.GetByOffset("0") << ".x);\n";
    } else {
      shader.MainFunctionBody() << "let b = " << b.GetByOffset("global_idx") << ";\n";
    }
  } else {
    const auto& c_indices = shader.AddIndices("bcast_indices");
    // check whether can use vectorize mode.
    // If either last dimension of A or B is divisible by 4, or the shared dimension is divisible by 4, vectorize mode
    // can be enabled.
    // In vectorize mode, the source data of A and B will be loaded only once to calculate 4 output values.
    // Use indices helpers to calculate the offset of A and B.
    if (vectorize_) {
      const auto& a_indices = shader.AddIndices("a_indices");
      const auto& b_indices = shader.AddIndices("b_indices");

      shader.MainFunctionBody() << "let outputIndices = " << c_indices.OffsetToIndices("global_idx * 4") << ";\n"
                                << "let offset_a = " << a_indices.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "let offset_b = " << b_indices.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n";
      // get A data
      if (a.NumComponents() == 4) {
        shader.MainFunctionBody() << "let a = " << a.GetByOffset("offset_a / 4") << ";\n";
      } else {
        shader.MainFunctionBody() << "let a = input_a_value_t(" << a.GetByOffset("offset_a") << ");\n";
      }

      // get B data
      if (b.NumComponents() == 4) {
        shader.MainFunctionBody() << "let b = " << b.GetByOffset("offset_b / 4") << ";\n";
      } else {
        shader.MainFunctionBody() << "let b = input_b_value_t(" << b.GetByOffset("offset_b") << ");\n";
      }
    } else {
      // In broadcast mode, each element of the vec4 value of A and B will be loaded separately to calculate the output value.
      shader.MainFunctionBody() << "var outputIndices = " << c_indices.OffsetToIndices("global_idx * 4") << ";\n"
                                << "let offset_a0 = " << a.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "let offset_b0 = " << b.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "outputIndices = " << c_indices.OffsetToIndices("global_idx * 4 + 1") << ";\n"
                                << "let offset_a1 = " << a.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "let offset_b1 = " << b.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "outputIndices = " << c_indices.OffsetToIndices("global_idx * 4 + 2") << ";\n"
                                << "let offset_a2 = " << a.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "let offset_b2 = " << b.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "outputIndices = " << c_indices.OffsetToIndices("global_idx * 4 + 3") << ";\n"
                                << "let offset_a3 = " << a.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "let offset_b3 = " << b.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n";

      // get A data
      shader.MainFunctionBody() << "let a = vec4<input_a_value_t>("
                                << a.GetByOffset("offset_a0") << ", "
                                << a.GetByOffset("offset_a1") << ", "
                                << a.GetByOffset("offset_a2") << ", "
                                << a.GetByOffset("offset_a3") << ");\n";
      // get B data
      shader.MainFunctionBody() << "let b = vec4<input_b_value_t>("
                                << b.GetByOffset("offset_b0") << ", "
                                << b.GetByOffset("offset_b1") << ", "
                                << b.GetByOffset("offset_b2") << ", "
                                << b.GetByOffset("offset_b3") << ");\n";
    }
  }

  shader.MainFunctionBody() << c.SetByOffset("global_idx", expression_);
  return Status::OK();
}

Status BinaryElementwise::ComputeInternal(ComputeContext& context) const {
  auto lhs_tensor = context.Input(0);
  auto rhs_tensor = context.Input(1);
  const auto& lhs_shape = lhs_tensor->Shape();
  const auto& rhs_shape = rhs_tensor->Shape();

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), lhs_shape, rhs_shape, output_shape));
  auto output_tensor = context.Output(0, output_shape);
  int64_t size = output_shape.Size();
  if (size == 0) {
    return Status::OK();
  }

  bool is_broadcast = lhs_shape != rhs_shape;
  bool is_lhs_scalar = lhs_shape.IsScalar();
  bool is_rhs_scalar = rhs_shape.IsScalar();

  bool vectorize = is_lhs_scalar || is_rhs_scalar || !is_broadcast;
  bool a_last_dim_divisible_by_4 = false;
  bool b_last_dim_divisible_by_4 = false;
  bool shared_dimension_divisible_by_4 = false;
  size_t num_shared_dimension = 0;
  if (!vectorize) {
    // check whether vectorize can be enabled
    a_last_dim_divisible_by_4 = lhs_shape.NumDimensions() > 0 && lhs_shape[lhs_shape.NumDimensions() - 1] % 4 == 0;
    b_last_dim_divisible_by_4 = rhs_shape.NumDimensions() > 0 && rhs_shape[rhs_shape.NumDimensions() - 1] % 4 == 0;
    if (a_last_dim_divisible_by_4 || b_last_dim_divisible_by_4) {
      vectorize = true;
    } else {
      int64_t shared_dimension = 1;
      for (size_t i = 1; i < output_shape.NumDimensions(); i++) {
        int64_t dimA = lhs_shape.NumDimensions() >= i ? lhs_shape[lhs_shape.NumDimensions() - i] : 1;
        int64_t dimB = rhs_shape.NumDimensions() >= i ? rhs_shape[rhs_shape.NumDimensions() - i] : 1;
        if (dimA == dimB) {
          shared_dimension *= dimA;
          num_shared_dimension++;
        } else {
          break;
        }
      }
      if (shared_dimension % 4 == 0) {
        shared_dimension_divisible_by_4 = true;
        vectorize = true;
      }
    }
  }

  uint32_t vec_size = gsl::narrow<uint32_t>((size + 3) / 4);
  BinaryElementwiseProgram program{kernel_name_,
                                   expression_,
                                   is_broadcast,
                                   is_lhs_scalar,
                                   is_rhs_scalar,
                                   vectorize};
  program
      .SetDispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          {static_cast<uint32_t>(vec_size)},
      })
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Type, {vec_size}, 4});

  if (is_lhs_scalar || is_rhs_scalar || !is_broadcast) {
    // Mode Element-wise
    // cache hint: "E{is_a_scalar}{is_b_scalar}"
    program
        .AddInputs({{lhs_tensor, ProgramTensorMetadataDependency::Type, {is_lhs_scalar ? 1 : vec_size}, 4},
                    {rhs_tensor, ProgramTensorMetadataDependency::Type, {is_rhs_scalar ? 1 : vec_size}, 4}})
        .CacheHint("E" + std::to_string(is_lhs_scalar) + std::to_string(is_rhs_scalar));
  } else if (vectorize) {
    // reshape the dims to merge the shared dimension if available
    bool need_reshape = shared_dimension_divisible_by_4 && num_shared_dimension > 1;
    TensorShape reshaped_lhs_shape = need_reshape ? lhs_shape.Slice(0, lhs_shape.NumDimensions() - num_shared_dimension + 1)
                                                  : lhs_shape;
    TensorShape reshaped_rhs_shape = need_reshape ? rhs_shape.Slice(0, rhs_shape.NumDimensions() - num_shared_dimension + 1)
                                                  : rhs_shape;
    TensorShape reshaped_output_shape = need_reshape ? output_shape.Slice(0, output_shape.NumDimensions() - num_shared_dimension + 1)
                                                     : output_shape;
    if (need_reshape) {
      reshaped_lhs_shape[reshaped_lhs_shape.NumDimensions() - 1] = lhs_shape.SizeFromDimension(lhs_shape.NumDimensions() - num_shared_dimension);
      reshaped_rhs_shape[reshaped_rhs_shape.NumDimensions() - 1] = rhs_shape.SizeFromDimension(rhs_shape.NumDimensions() - num_shared_dimension);
      reshaped_output_shape[reshaped_output_shape.NumDimensions() - 1] = output_shape.SizeFromDimension(output_shape.NumDimensions() - num_shared_dimension);
    }

    if (shared_dimension_divisible_by_4 || a_last_dim_divisible_by_4) {
      program.AddInput({lhs_tensor, ProgramTensorMetadataDependency::Type, {(lhs_shape.Size() + 3) / 4}, 4});
    } else {
      program.AddInput({lhs_tensor, ProgramTensorMetadataDependency::Type});
    }
    if (shared_dimension_divisible_by_4 || b_last_dim_divisible_by_4) {
      program.AddInput({rhs_tensor, ProgramTensorMetadataDependency::Type, {(rhs_shape.Size() + 3) / 4}, 4});
    } else {
      program.AddInput({rhs_tensor, ProgramTensorMetadataDependency::Type});
    }
    // Mode Vectorize broadcast
    // cache hint: "V{a_rank};{b_rank};{output_rank}"
    program
        .AddIndices(reshaped_output_shape)
        .AddIndices(reshaped_lhs_shape)
        .AddIndices(reshaped_rhs_shape)
        .CacheHint("V");
  } else {
    // Mode Broadcast
    // cache hint: "B"
    program
        .AddInputs({{lhs_tensor, ProgramTensorMetadataDependency::TypeAndRank},
                    {rhs_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
        .AddIndices(output_tensor->Shape())
        .CacheHint("B");
  }

  return context.RunProgram(program);
}

#define WEBGPU_BINARY_IMPL(OP_TYPE, ...)                                                  \
  class OP_TYPE final : public BinaryElementwise {                                        \
   public:                                                                                \
    OP_TYPE(const OpKernelInfo& info) : BinaryElementwise{info, #OP_TYPE, __VA_ARGS__} {} \
  };

#define WEBGPU_BINARY_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_KERNEL_EX(                                         \
      OP_TYPE,                                                     \
      kOnnxDomain,                                                 \
      VERSION,                                                     \
      kWebGpuExecutionProvider,                                    \
      KernelDefBuilder().TypeConstraint("T", TYPE),                \
      KERNEL_CLASS);

#define WEBGPU_BINARY_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                          \
      OP_TYPE,                                                                                \
      kOnnxDomain,                                                                            \
      VERSION_FROM, VERSION_TO,                                                               \
      kWebGpuExecutionProvider,                                                               \
      KernelDefBuilder().TypeConstraint("T", TYPE),                                           \
      KERNEL_CLASS);

#define WEBGPU_BINARY_KERNEL_2(OP_TYPE, VERSION, KERNEL_CLASS, TYPE, TYPE1) \
  ONNX_OPERATOR_KERNEL_EX(                                                  \
      OP_TYPE,                                                              \
      kOnnxDomain,                                                          \
      VERSION,                                                              \
      kWebGpuExecutionProvider,                                             \
      KernelDefBuilder()                                                    \
          .TypeConstraint("T", TYPE)                                        \
          .TypeConstraint("T1", TYPE1),                                     \
      KERNEL_CLASS);

#define WEBGPU_BINARY_VERSIONED_KERNEL_2(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE, TYPE1) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                                   \
      OP_TYPE,                                                                                         \
      kOnnxDomain,                                                                                     \
      VERSION_FROM, VERSION_TO,                                                                        \
      kWebGpuExecutionProvider,                                                                        \
      KernelDefBuilder()                                                                               \
          .TypeConstraint("T", TYPE)                                                                   \
          .TypeConstraint("T1", TYPE1),                                                                \
      KERNEL_CLASS);

WEBGPU_BINARY_IMPL(Add, "a + b")
WEBGPU_BINARY_VERSIONED_KERNEL(Add, 7, 12, Add, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL(Add, 13, 13, Add, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(Add, 14, Add, WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(Div, "a / b")
WEBGPU_BINARY_VERSIONED_KERNEL(Div, 7, 12, Div, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL(Div, 13, 13, Div, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(Div, 14, Div, WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(Mul, "a * b")
WEBGPU_BINARY_VERSIONED_KERNEL(Mul, 7, 12, Mul, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL(Mul, 13, 13, Mul, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(Mul, 14, Mul, WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(Sub, "a - b")
WEBGPU_BINARY_VERSIONED_KERNEL(Sub, 7, 12, Sub, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL(Sub, 13, 13, Sub, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(Sub, 14, Sub, WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(Pow, "output_value_t(pow(vec4<f32>(a), vec4<f32>(b)))")
WEBGPU_BINARY_VERSIONED_KERNEL(Pow, 7, 11, Pow, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL_2(Pow, 12, 12, Pow, WebGpuSupportedNumberTypes(), WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL_2(Pow, 13, 14, Pow, WebGpuSupportedNumberTypes(), WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL_2(Pow, 15, Pow, WebGpuSupportedNumberTypes(), WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(Equal, "vec4<u32>(a == b)")
WEBGPU_BINARY_VERSIONED_KERNEL(Equal, 7, 10, Equal, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL(Equal, 11, 12, Equal, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL(Equal, 13, 18, Equal, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(Equal, 19, Equal, WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(Greater, "vec4<u32>(a > b)")
WEBGPU_BINARY_VERSIONED_KERNEL(Greater, 7, 8, Greater, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL(Greater, 9, 12, Greater, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(Greater, 13, Greater, WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(Less, "vec4<u32>(a < b)")
WEBGPU_BINARY_VERSIONED_KERNEL(Less, 7, 8, Less, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL(Less, 9, 12, Less, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(Less, 13, Less, WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(GreaterOrEqual, "vec4<u32>(a >= b)")
WEBGPU_BINARY_VERSIONED_KERNEL(GreaterOrEqual, 12, 15, GreaterOrEqual, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(GreaterOrEqual, 16, GreaterOrEqual, WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(LessOrEqual, "vec4<u32>(a <= b)")
WEBGPU_BINARY_VERSIONED_KERNEL(LessOrEqual, 12, 15, LessOrEqual, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(LessOrEqual, 16, LessOrEqual, WebGpuSupportedNumberTypes())

}  // namespace webgpu
}  // namespace onnxruntime
