// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>

#include "core/common/inlined_containers.h"
#include "core/providers/common.h"
#include "core/providers/webgpu/math/binary_elementwise_ops.h"
#include "core/providers/webgpu/math/binary_elementwise_broadcast_utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/string_macros.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status BinaryElementwiseProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("input_a", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& b = shader.AddInput("input_b", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& c = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  const bool a_is_bool = Inputs()[0].var_type == ProgramVariableDataType::Boolx4;
  const bool b_is_bool = Inputs()[1].var_type == ProgramVariableDataType::Boolx4;

  shader.AdditionalImplementation() << additional_impl_;

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size");

  if (is_int64_input_ || is_int64_output_) {
    // INT64 input or output (component=1): declare shared base and element_count for use in the shader.
    shader.MainFunctionBody()
        << "let base = global_idx * 4u;\n"
        << "let element_count = uniforms.element_count;\n";
  }

  // check whether can use element-wise mode.
  // If either A or B is scalar, or A and B have the same shape, element-wise mode can be used.
  // In element-wise mode, no indices calculation is needed.
  if (is_lhs_scalar_ || is_rhs_scalar_ || !is_broadcast_) {
    if (is_int64_input_) {
      // INT64 inputs have component=1; read 4 individual elements into vec4 for uniform processing.
      // Guard lanes 1-3 against OOB reads when size is not divisible by 4.
      const auto a_offset = [&](const std::string& idx) {
        return is_lhs_scalar_ ? a.GetByOffset("0") : a.GetByOffset(idx);
      };
      const auto b_offset = [&](const std::string& idx) {
        return is_rhs_scalar_ ? b.GetByOffset("0") : b.GetByOffset(idx);
      };
      shader.MainFunctionBody()
          << "var a0 = " << a_offset("base") << ";\n"
          << "var b0 = " << b_offset("base") << ";\n"
          << "var a1 = input_a_value_t(0); var a2 = input_a_value_t(0); var a3 = input_a_value_t(0);\n"
          << "var b1 = input_b_value_t(0); var b2 = input_b_value_t(0); var b3 = input_b_value_t(0);\n"
          << "if (base + 1u < element_count) { a1 = " << a_offset("base + 1u") << "; b1 = " << b_offset("base + 1u") << "; }\n"
          << "if (base + 2u < element_count) { a2 = " << a_offset("base + 2u") << "; b2 = " << b_offset("base + 2u") << "; }\n"
          << "if (base + 3u < element_count) { a3 = " << a_offset("base + 3u") << "; b3 = " << b_offset("base + 3u") << "; }\n"
          << "let a = vec4<input_a_value_t>(a0, a1, a2, a3);\n"
          << "let b = vec4<input_b_value_t>(b0, b1, b2, b3);\n";
    } else {
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
    }
  } else {
    const auto& c_indices = shader.AddIndices("bcast_indices");
    // Use indices helpers to calculate the offset of A and B.
    const auto& a_indices = shader.AddIndices("a_indices");
    const auto& b_indices = shader.AddIndices("b_indices");

    // check whether can use vectorize mode.
    // If either last dimension of A or B is divisible by 4, or the shared dimension is divisible by 4, vectorize mode
    // can be enabled.
    // In vectorize mode, the source data of A and B will be loaded only once to calculate 4 output values.
    if (vectorize_) {
      shader.MainFunctionBody() << "let outputIndices = " << c_indices.OffsetToIndices("global_idx * 4") << ";\n"
                                << "let offset_a = " << a_indices.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "let offset_b = " << b_indices.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n";
      // get A data
      if (is_lhs_use_4_components_) {
        shader.MainFunctionBody() << "let a = " << a.GetByOffset("offset_a / 4") << ";\n";
      } else if (a_is_bool) {
        shader.MainFunctionBody() << "let a = " << a.GetByOffset("offset_a / 4") << "[offset_a % 4];\n";
      } else {
        shader.MainFunctionBody() << "let a = input_a_value_t(" << a.GetByOffset("offset_a") << ");\n";
      }

      // get B data
      if (is_rhs_use_4_components_) {
        shader.MainFunctionBody() << "let b = " << b.GetByOffset("offset_b / 4") << ";\n";
      } else if (b_is_bool) {
        shader.MainFunctionBody() << "let b = " << b.GetByOffset("offset_b / 4") << "[offset_b % 4];\n";
      } else {
        shader.MainFunctionBody() << "let b = input_b_value_t(" << b.GetByOffset("offset_b") << ");\n";
      }
    } else {
      // In broadcast mode, each element of the vec4 value of A and B will be loaded separately to calculate the output value.
      shader.MainFunctionBody() << "var outputIndices = " << c_indices.OffsetToIndices("global_idx * 4") << ";\n"
                                << "let offset_a0 = " << a_indices.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "let offset_b0 = " << b_indices.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "outputIndices = " << c_indices.OffsetToIndices("global_idx * 4 + 1") << ";\n"
                                << "let offset_a1 = " << a_indices.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "let offset_b1 = " << b_indices.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "outputIndices = " << c_indices.OffsetToIndices("global_idx * 4 + 2") << ";\n"
                                << "let offset_a2 = " << a_indices.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "let offset_b2 = " << b_indices.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "outputIndices = " << c_indices.OffsetToIndices("global_idx * 4 + 3") << ";\n"
                                << "let offset_a3 = " << a_indices.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n"
                                << "let offset_b3 = " << b_indices.BroadcastedIndicesToOffset("outputIndices", c_indices) << ";\n";

      // get A data
      if (a_is_bool) {
        shader.MainFunctionBody() << "let a = vec4<bool>("
                                  << a.GetByOffset("offset_a0 / 4") << "[offset_a0 % 4], "
                                  << a.GetByOffset("offset_a1 / 4") << "[offset_a1 % 4], "
                                  << a.GetByOffset("offset_a2 / 4") << "[offset_a2 % 4], "
                                  << a.GetByOffset("offset_a3 / 4") << "[offset_a3 % 4]);\n";
      } else {
        shader.MainFunctionBody() << "let a = vec4<input_a_value_t>("
                                  << a.GetByOffset("offset_a0") << ", "
                                  << a.GetByOffset("offset_a1") << ", "
                                  << a.GetByOffset("offset_a2") << ", "
                                  << a.GetByOffset("offset_a3") << ");\n";
      }
      // get B data
      if (b_is_bool) {
        shader.MainFunctionBody() << "let b = vec4<bool>("
                                  << b.GetByOffset("offset_b0 / 4") << "[offset_b0 % 4], "
                                  << b.GetByOffset("offset_b1 / 4") << "[offset_b1 % 4], "
                                  << b.GetByOffset("offset_b2 / 4") << "[offset_b2 % 4], "
                                  << b.GetByOffset("offset_b3 / 4") << "[offset_b3 % 4]);\n";
      } else {
        shader.MainFunctionBody() << "let b = vec4<input_b_value_t>("
                                  << b.GetByOffset("offset_b0") << ", "
                                  << b.GetByOffset("offset_b1") << ", "
                                  << b.GetByOffset("offset_b2") << ", "
                                  << b.GetByOffset("offset_b3") << ");\n";
      }
    }
  }

  if (is_int64_output_) {
    // INT64 output (component=1): write each component of the vec4 result individually.
    shader.MainFunctionBody()
        << "let result = " << expression_ << ";\n"
        << c.SetByOffset("base", "result[0]") << "\n"
        << "if (base + 1u < element_count) { " << c.SetByOffset("base + 1u", "result[1]") << " }\n"
        << "if (base + 2u < element_count) { " << c.SetByOffset("base + 2u", "result[2]") << " }\n"
        << "if (base + 3u < element_count) { " << c.SetByOffset("base + 3u", "result[3]") << " }\n";
  } else {
    shader.MainFunctionBody() << c.SetByOffset("global_idx", expression_);
  }
  return Status::OK();
}

namespace {
// Builds and runs a BinaryElementwiseProgram that computes `output = expression(lhs, rhs)`
// with multidirectional broadcasting. The output tensor must already be sized to the broadcast
// shape of `lhs` and `rhs`, and must contain at least one element.
Status RunBinaryProgram(ComputeContext& context,
                        const std::string& kernel_name,
                        const std::string& expression,
                        const std::string& additional_impl,
                        const Tensor* lhs_tensor,
                        const Tensor* rhs_tensor,
                        Tensor* output_tensor) {
  const auto& lhs_shape = lhs_tensor->Shape();
  const auto& rhs_shape = rhs_tensor->Shape();
  const auto& output_shape = output_tensor->Shape();
  int64_t size = output_shape.Size();

  bool is_broadcast = lhs_shape != rhs_shape;
  bool is_lhs_scalar = lhs_shape.IsScalar();
  bool is_rhs_scalar = rhs_shape.IsScalar();

  // Check if either input is boolean
  // For boolean inputs, we need to handle them differently in the shader. This is because `bool` is not a valid type in
  // storage buffer. We have to use a `u32` to represent 4 boolean values.
  bool is_lhs_bool = lhs_tensor->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  bool is_rhs_bool = rhs_tensor->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;

  // INT64 has no vec4 representation in WebGPU (stored as vec2<u32>), so disable vectorization.
  bool is_lhs_int64 = lhs_tensor->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  bool is_rhs_int64 = rhs_tensor->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  bool is_int64_input = is_lhs_int64 || is_rhs_int64;

  bool vectorize = !is_int64_input && (is_lhs_scalar || is_rhs_scalar || !is_broadcast);
  bool a_last_dim_divisible_by_4 = false;
  bool b_last_dim_divisible_by_4 = false;
  bool shared_dimension_divisible_by_4 = false;
  size_t num_shared_dimension = 0;
  if (!is_int64_input && !vectorize) {
    // check whether vectorize can be enabled
    a_last_dim_divisible_by_4 = lhs_shape.NumDimensions() > 0 && lhs_shape[lhs_shape.NumDimensions() - 1] % 4 == 0;
    b_last_dim_divisible_by_4 = rhs_shape.NumDimensions() > 0 && rhs_shape[rhs_shape.NumDimensions() - 1] % 4 == 0;
    if (a_last_dim_divisible_by_4 || b_last_dim_divisible_by_4) {
      vectorize = true;
    } else {
      int64_t shared_dimension = 1;
      num_shared_dimension = CountSharedTrailingDimensions(lhs_shape, rhs_shape,
                                                           output_shape.NumDimensions(), shared_dimension);
      if (shared_dimension % 4 == 0) {
        shared_dimension_divisible_by_4 = true;
        vectorize = true;
      }
    }
  }

  bool is_int64_output = output_tensor->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  uint32_t vec_size = onnxruntime::narrow<uint32_t>((size + 3) / 4);
  int output_component = is_int64_output ? 1 : 4;
  uint32_t output_size = is_int64_output ? onnxruntime::narrow<uint32_t>(size) : vec_size;

  BinaryElementwiseProgram program{kernel_name,
                                   expression,
                                   additional_impl,
                                   is_broadcast,
                                   is_lhs_scalar,
                                   is_rhs_scalar,
                                   shared_dimension_divisible_by_4 || a_last_dim_divisible_by_4,
                                   shared_dimension_divisible_by_4 || b_last_dim_divisible_by_4,
                                   vectorize,
                                   is_int64_input,
                                   is_int64_output};
  program
      .SetDispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          {static_cast<uint32_t>(vec_size)},
          {static_cast<uint32_t>(size)},
      })
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Type, {output_size}, output_component});

  if (is_lhs_scalar || is_rhs_scalar || !is_broadcast) {
    // Mode Element-wise
    // cache hint: "E{is_a_scalar}{is_b_scalar}"
    program
        .AddInputs({{lhs_tensor, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, is_int64_input ? 1 : 4},
                    {rhs_tensor, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, is_int64_input ? 1 : 4}})
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

    if (shared_dimension_divisible_by_4 || a_last_dim_divisible_by_4 || is_lhs_bool) {
      program.AddInput({lhs_tensor, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4});
    } else {
      program.AddInput({lhs_tensor, ProgramTensorMetadataDependency::Type});
    }
    if (shared_dimension_divisible_by_4 || b_last_dim_divisible_by_4 || is_rhs_bool) {
      program.AddInput({rhs_tensor, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4});
    } else {
      program.AddInput({rhs_tensor, ProgramTensorMetadataDependency::Type});
    }
    // Mode Vectorize broadcast
    // cache hint: "V{a_rank};{b_rank};{output_rank}"
    program
        .AddIndices(std::move(reshaped_output_shape))
        .AddIndices(std::move(reshaped_lhs_shape))
        .AddIndices(std::move(reshaped_rhs_shape))
        .CacheHint("V");
  } else {
    // Mode Broadcast
    // cache hint: "B"
    program
        .AddInputs({{lhs_tensor, ProgramTensorMetadataDependency::TypeAndRank, ProgramInput::Flatten, is_lhs_bool ? 4 : 1},
                    {rhs_tensor, ProgramTensorMetadataDependency::TypeAndRank, ProgramInput::Flatten, is_rhs_bool ? 4 : 1}})
        .AddIndices(output_tensor->Shape())
        .AddIndices(lhs_tensor->Shape())
        .AddIndices(rhs_tensor->Shape())
        .CacheHint("B");
  }

  return context.RunProgram(program);
}
}  // namespace

Status BinaryElementwise::ComputeInternal(ComputeContext& context) const {
  auto lhs_tensor = context.Input(0);
  auto rhs_tensor = context.Input(1);
  const auto& lhs_shape = lhs_tensor->Shape();
  const auto& rhs_shape = rhs_tensor->Shape();

  TensorShape output_shape;
  ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), lhs_shape, rhs_shape, output_shape));
  auto output_tensor = context.Output(0, output_shape);
  if (output_shape.Size() == 0) {
    return Status::OK();
  }

  std::string additional_impl;
  if (get_additional_impl_) {
    additional_impl = get_additional_impl_(lhs_tensor->GetElementType(), rhs_tensor->GetElementType());
  }

  return RunBinaryProgram(context, kernel_name_, expression_, additional_impl, lhs_tensor, rhs_tensor, output_tensor);
}

Status VariadicElementwise::ComputeInternal(ComputeContext& context) const {
  const int input_count = context.InputCount();
  const auto* input_0 = context.Input(0);

  // Single input: the output is a copy of the input.
  if (input_count == 1) {
    auto* output_tensor = context.Output(0, input_0->Shape());
    if (output_tensor->Shape().Size() == 0) {
      return Status::OK();
    }
    return context.CopyTensor(*input_0, *output_tensor);
  }

  // Compute the multidirectional (NumPy-style) broadcast output shape across all inputs.
  TensorShape output_shape = input_0->Shape();
  for (int i = 1; i < input_count; ++i) {
    TensorShape accumulated_shape = output_shape;
    ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), accumulated_shape, context.Input(i)->Shape(), output_shape));
  }
  auto* output_tensor = context.Output(0, output_shape);
  if (output_shape.Size() == 0) {
    return Status::OK();
  }

  // Fold the inputs pairwise: acc = op(acc, input[i]).
  // Intermediate results (for input_count > 2) are held in temporary GPU tensors that must stay
  // alive until their consuming program has been queued, so they are kept in a vector for the
  // duration of this call. Reserve up front so the vector never reallocates and invalidates the
  // pointers handed to the next iteration.
  const auto element_type = input_0->DataType();
  std::string additional_impl;
  if (get_additional_impl_) {
    additional_impl = get_additional_impl_(input_0->GetElementType(), input_0->GetElementType());
  }
  InlinedVector<Tensor> intermediate_tensors;
  // input_count >= 2 here (the single-input case returned above), so the last fold targets the
  // kernel output and there are input_count - 2 intermediates. Guard the subtraction anyway so a
  // future refactor can't turn it into an unsigned underflow.
  if (input_count > 2) {
    intermediate_tensors.reserve(static_cast<size_t>(input_count) - 2);
  }

  const Tensor* lhs_tensor = input_0;
  for (int i = 1; i < input_count; ++i) {
    const Tensor* rhs_tensor = context.Input(i);
    Tensor* dst_tensor = nullptr;
    if (i == input_count - 1) {
      // The last fold writes directly to the kernel output.
      dst_tensor = output_tensor;
    } else {
      TensorShape intermediate_shape;
      ORT_RETURN_IF_ERROR(ComputeBroadcastOutputShape(Node().Name(), lhs_tensor->Shape(), rhs_tensor->Shape(), intermediate_shape));
      intermediate_tensors.push_back(context.CreateGPUTensor(element_type, intermediate_shape));
      dst_tensor = &intermediate_tensors.back();
    }
    ORT_RETURN_IF_ERROR(RunBinaryProgram(context, kernel_name_, expression_, additional_impl,
                                         lhs_tensor, rhs_tensor, dst_tensor));
    lhs_tensor = dst_tensor;
  }

  return Status::OK();
}

#define WEBGPU_BINARY_IMPL(OP_TYPE, ...)                                                  \
  class OP_TYPE final : public BinaryElementwise {                                        \
   public:                                                                                \
    OP_TYPE(const OpKernelInfo& info) : BinaryElementwise{info, #OP_TYPE, __VA_ARGS__} {} \
  };

#define WEBGPU_VARIADIC_IMPL(OP_TYPE, ...)                                                  \
  class OP_TYPE final : public VariadicElementwise {                                        \
   public:                                                                                  \
    OP_TYPE(const OpKernelInfo& info) : VariadicElementwise{info, #OP_TYPE, __VA_ARGS__} {} \
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

// NOTE: int64 arithmetic in the WebGPU shader operates on the low 32 bits only (i32 element type).
// Values outside the int32 range [-2^31, 2^31-1] will produce incorrect results.
// This matches the same limitation documented in Range and is acceptable for token-position workloads.
template <int StartVersion, int EndVersion>
KernelCreateInfo CreateSubVersionedKernelInfo(bool enable_int64) {
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, false);
  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Sub>(info);
    return Status::OK();
  };
  return {KernelDefBuilder()
              .SetName("Sub")
              .SetDomain(kOnnxDomain)
              .SinceVersion(StartVersion, EndVersion)
              .Provider(kWebGpuExecutionProvider)
              .TypeConstraint("T", type_constraints)
              .Build(),
          kernel_create_fn};
}

template <int SinceVersion>
KernelCreateInfo CreateSubKernelInfo(bool enable_int64) {
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, false);
  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Sub>(info);
    return Status::OK();
  };
  return {KernelDefBuilder()
              .SetName("Sub")
              .SetDomain(kOnnxDomain)
              .SinceVersion(SinceVersion)
              .Provider(kWebGpuExecutionProvider)
              .TypeConstraint("T", type_constraints)
              .Build(),
          kernel_create_fn};
}

template KernelCreateInfo CreateSubVersionedKernelInfo<7, 12>(bool);
template KernelCreateInfo CreateSubVersionedKernelInfo<13, 13>(bool);
template KernelCreateInfo CreateSubKernelInfo<14>(bool);

// ONNX Max/Min (opset 12+) propagate NaN: if either operand is NaN the result is NaN.
// The WGSL `max`/`min` builtins do not guarantee this, so wrap them so that a NaN operand is
// forwarded. NaN is detected via an integer bitcast (NaN iff `(bits & 0x7fffffff) > 0x7f800000`)
// rather than the `x != x` self-inequality: shader compilers assume floats are never NaN and fold
// `x != x` to `false`, which silently breaks propagation. Integer bit math is not subject to that
// fast-math assumption. f16 is widened to f32 first (NaN is preserved) so the 16-byte
// `vec4<u32>` bitcast is valid. For integer element types no NaN handling is emitted.
static std::string GetMinMaxImpl(int element_type, bool is_max) {
  const char* fn_name = is_max ? "max_v" : "min_v";
  const char* builtin = is_max ? "max" : "min";
  SS(s, 1024);
  const bool is_float = element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                        element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  s << "fn " << fn_name << "(a : vec4<input_a_element_t>, b : vec4<input_b_element_t>) -> vec4<input_a_element_t> {\n";
  if (is_float) {
    const bool is_f16 = element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    const std::string a_bits = is_f16 ? "bitcast<vec4<u32>>(vec4<f32>(a))" : "bitcast<vec4<u32>>(a)";
    const std::string b_bits = is_f16 ? "bitcast<vec4<u32>>(vec4<f32>(b))" : "bitcast<vec4<u32>>(b)";
    s << "  let a_nan = (" << a_bits << " & vec4<u32>(0x7fffffffu)) > vec4<u32>(0x7f800000u);\n"
      << "  let b_nan = (" << b_bits << " & vec4<u32>(0x7fffffffu)) > vec4<u32>(0x7f800000u);\n"
      << "  return select(select(" << builtin << "(a, b), b, b_nan), a, a_nan);\n";
  } else {
    s << "  return " << builtin << "(a, b);\n";
  }
  s << "}\n";
  return SS_GET(s);
}

static std::string GetMaxImpl(int lhs_element_type, int /* rhs_element_type */) {
  return GetMinMaxImpl(lhs_element_type, /*is_max=*/true);
}
static std::string GetMinImpl(int lhs_element_type, int /* rhs_element_type */) {
  return GetMinMaxImpl(lhs_element_type, /*is_max=*/false);
}

WEBGPU_VARIADIC_IMPL(Max, "max_v(vec4<input_a_element_t>(a), vec4<input_b_element_t>(b))", GetMaxImpl)
WEBGPU_BINARY_VERSIONED_KERNEL(Max, 8, 11, Max, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL(Max, 12, 12, Max, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(Max, 13, Max, WebGpuSupportedNumberTypes())

WEBGPU_VARIADIC_IMPL(Min, "min_v(vec4<input_a_element_t>(a), vec4<input_b_element_t>(b))", GetMinImpl)
WEBGPU_BINARY_VERSIONED_KERNEL(Min, 8, 11, Min, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL(Min, 12, 12, Min, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(Min, 13, Min, WebGpuSupportedNumberTypes())

std::string GetPowImpl(int lhs_element_type, int /* rhs_element_type */) {
  SS(s, 1024);
  std::string round_str;
  if (lhs_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    round_str = "round";
  }
  std::string use_pow_shortcut;
  if (lhs_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || lhs_element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    // use multiplication instead of pow when base (a) is a float and exponent (b) is 2.0
    // use sqrt instead of pow when base (a) is a positive float and exponent (b) is 0.5
    use_pow_shortcut =
        "  else if (b == 2.0) {\n"
        "    return a * a;\n"
        "  } else if (a >= input_a_element_t(0.0) && b == 0.5) {\n"
        "    return sqrt(a);\n"
        "  }\n";
  }

  s << "fn pow_custom(a : input_a_element_t, b : f32) -> input_a_element_t {\n"
       "  if (b == 0.0) {\n"
       "    return input_a_element_t(1.0);\n"
       "  } else if (a < input_a_element_t(0.0) && b != floor(b)) {\n"
       "    return input_a_element_t(pow(f32(a), b)); // NaN\n"
       "  }\n"
    << use_pow_shortcut
    << "  return select(sign(a), input_a_element_t(1.0), round(abs(b) % 2.0) != 1.0) * input_a_element_t(" << round_str << "(pow(f32(abs(a)), b)));\n"
    << "}\n"
       "fn pow_v(a : vec4<input_a_element_t>, b : vec4<input_b_element_t>) -> vec4<input_a_element_t> {\n"
       "  return vec4<input_a_element_t>(pow_custom(a.x, f32(b.x)), pow_custom(a.y, f32(b.y)), pow_custom(a.z, f32(b.z)), pow_custom(a.w, f32(b.w)));\n"
       "}\n";
  return SS_GET(s);
}

WEBGPU_BINARY_IMPL(Pow, "pow_v(a, b)", GetPowImpl)
WEBGPU_BINARY_VERSIONED_KERNEL(Pow, 7, 11, Pow, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL_2(Pow, 12, 12, Pow, WebGpuSupportedNumberTypes(), WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL_2(Pow, 13, 14, Pow, WebGpuSupportedNumberTypes(), WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL_2(Pow, 15, Pow, WebGpuSupportedNumberTypes(), WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(Equal, "vec4<u32>(vec4<input_a_element_t>(a) == vec4<input_b_element_t>(b))")

// NOTE: int64 comparison in the WebGPU shader uses i32 element type (low 32 bits only).
// Values outside the int32 range will produce incorrect results.
template <int StartVersion, int EndVersion>
KernelCreateInfo CreateEqualVersionedKernelInfo(bool enable_int64) {
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, false);
  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Equal>(info);
    return Status::OK();
  };
  return {KernelDefBuilder()
              .SetName("Equal")
              .SetDomain(kOnnxDomain)
              .SinceVersion(StartVersion, EndVersion)
              .Provider(kWebGpuExecutionProvider)
              .TypeConstraint("T", type_constraints)
              .Build(),
          kernel_create_fn};
}

template <int SinceVersion>
KernelCreateInfo CreateEqualKernelInfo(bool enable_int64) {
  const auto& type_constraints = GetOpTypeConstraints(enable_int64, false);
  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Equal>(info);
    return Status::OK();
  };
  return {KernelDefBuilder()
              .SetName("Equal")
              .SetDomain(kOnnxDomain)
              .SinceVersion(SinceVersion)
              .Provider(kWebGpuExecutionProvider)
              .TypeConstraint("T", type_constraints)
              .Build(),
          kernel_create_fn};
}

template KernelCreateInfo CreateEqualVersionedKernelInfo<7, 10>(bool);
template KernelCreateInfo CreateEqualVersionedKernelInfo<11, 12>(bool);
template KernelCreateInfo CreateEqualVersionedKernelInfo<13, 18>(bool);
template KernelCreateInfo CreateEqualKernelInfo<19>(bool);

WEBGPU_BINARY_IMPL(Greater, "vec4<u32>(vec4<input_a_element_t>(a) > vec4<input_b_element_t>(b))")
WEBGPU_BINARY_VERSIONED_KERNEL(Greater, 7, 8, Greater, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL(Greater, 9, 12, Greater, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(Greater, 13, Greater, WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(Less, "vec4<u32>(vec4<input_a_element_t>(a) < vec4<input_b_element_t>(b))")
WEBGPU_BINARY_VERSIONED_KERNEL(Less, 7, 8, Less, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_VERSIONED_KERNEL(Less, 9, 12, Less, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(Less, 13, Less, WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(GreaterOrEqual, "vec4<u32>(vec4<input_a_element_t>(a) >= vec4<input_b_element_t>(b))")
WEBGPU_BINARY_VERSIONED_KERNEL(GreaterOrEqual, 12, 15, GreaterOrEqual, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(GreaterOrEqual, 16, GreaterOrEqual, WebGpuSupportedNumberTypes())

WEBGPU_BINARY_IMPL(LessOrEqual, "vec4<u32>(vec4<input_a_element_t>(a) <= vec4<input_b_element_t>(b))")
WEBGPU_BINARY_VERSIONED_KERNEL(LessOrEqual, 12, 15, LessOrEqual, WebGpuSupportedNumberTypes())
WEBGPU_BINARY_KERNEL(LessOrEqual, 16, LessOrEqual, WebGpuSupportedNumberTypes())

// And operator only supports tensor(bool).
WEBGPU_BINARY_IMPL(And, "(vec4<input_a_element_t>(a) & vec4<input_b_element_t>(b))")
WEBGPU_BINARY_KERNEL(And, 7, And, DataTypeImpl::GetTensorType<bool>())

}  // namespace webgpu
}  // namespace onnxruntime
