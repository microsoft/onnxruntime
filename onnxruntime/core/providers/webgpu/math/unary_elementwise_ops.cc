// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <utility>
#include <cstring>
#include <limits>

#include "core/providers/webgpu/math/unary_elementwise_ops.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {
Status UnaryElementwiseProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("x", ShaderUsage::UseUniform | additional_usage_);
  const auto& output = shader.AddOutput("y", ShaderUsage::UseUniform);
  shader.AdditionalImplementation() << additional_impl_;
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size")
                            << "  let a = " << input.GetByOffset("global_idx") << ";\n  "
                            << output.SetByOffset("global_idx", expression_);

  return Status::OK();
}

Status UnaryElementwise::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  auto* output_tensor = context.Output(0, input_tensor->Shape());
  int64_t size = input_tensor->Shape().Size();
  if (size == 0) {
    return Status::OK();
  }
  uint32_t vec_size = onnxruntime::narrow<uint32_t>((size + 3) / 4);
  UnaryElementwiseProgram program{kernel_name_, expression_, additional_impl_, additional_usage_};
  program
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::Type, {vec_size}, 4}})
      .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::None, {vec_size}, 4}})
      .SetDispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          {static_cast<uint32_t>(vec_size)},
      });
  if (!cache_hint.empty()) {
    program.CacheHint(cache_hint);
  }
  ORT_RETURN_IF_ERROR(ConfigureProgram(context, program));
  return context.RunProgram(program);
}

#define WEBGPU_ELEMENTWISE_IMPL(OP_TYPE, ...)                                            \
  class OP_TYPE final : public UnaryElementwise {                                        \
   public:                                                                               \
    OP_TYPE(const OpKernelInfo& info) : UnaryElementwise{info, #OP_TYPE, __VA_ARGS__} {} \
  };

#define WEBGPU_ELEMENTWISE_KERNEL(OP_TYPE_AND_CLASS_NAME, VERSION, TYPE)      \
  ONNX_OPERATOR_KERNEL_EX(                                                    \
      OP_TYPE_AND_CLASS_NAME, kOnnxDomain, VERSION, kWebGpuExecutionProvider, \
      KernelDefBuilder().TypeConstraint("T", TYPE),                           \
      OP_TYPE_AND_CLASS_NAME);

#define WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(OP_TYPE_AND_CLASS_NAME, VERSION_FROM, VERSION_TO, TYPE) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                                \
      OP_TYPE_AND_CLASS_NAME, kOnnxDomain, VERSION_FROM, VERSION_TO, kWebGpuExecutionProvider,      \
      KernelDefBuilder().TypeConstraint("T", TYPE),                                                 \
      OP_TYPE_AND_CLASS_NAME);

#define WEBGPU_ELEMENTWISE_BOOLEAN_KERNEL(OP_TYPE_AND_CLASS_NAME, VERSION)         \
  ONNX_OPERATOR_KERNEL_EX(                                                         \
      OP_TYPE_AND_CLASS_NAME, kOnnxDomain, VERSION, kWebGpuExecutionProvider,      \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<bool>()), \
      OP_TYPE_AND_CLASS_NAME);

//
// math
//

WEBGPU_ELEMENTWISE_IMPL(Abs, "abs(a)")
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Abs, 6, 12, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Abs, 13, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Neg, "-a")
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Neg, 6, 12, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Neg, 13, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Floor, "floor(a)")
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Floor, 6, 12, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Floor, 13, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Ceil, "ceil(a)")
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Ceil, 6, 12, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Ceil, 13, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Reciprocal, "1.0/a")
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Reciprocal, 6, 12, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Reciprocal, 13, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Sqrt, "sqrt(a)")
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Sqrt, 6, 12, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Sqrt, 13, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Exp, "exp(a)")
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Exp, 6, 12, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Exp, 13, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Erf, "erf_v(a)", ErfImpl, ShaderUsage::UseValueTypeAlias)
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Erf, 9, 12, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Erf, 13, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Log, "log(a)")
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Log, 6, 12, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Log, 13, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Sigmoid, "1.0 / (1.0 + exp(-a))")
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Sigmoid, 6, 12, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Sigmoid, 13, WebGpuSupportedFloatTypes())

class HardSigmoid final : public UnaryElementwise {
 public:
  HardSigmoid(const OpKernelInfo& info)
      : UnaryElementwise{info, "HardSigmoid", "hard_sigmoid_v(a)", HardSigmoidImpl, ShaderUsage::UseElementTypeAlias} {
    // attr[0] is alpha, attr[1] is beta
    info.GetAttrOrDefault("alpha", attr, 0.2f);
    info.GetAttrOrDefault("beta", attr + 1, 0.5f);
  }

  Status ConfigureProgram(const ComputeContext& /*context*/, UnaryElementwiseProgram& program) const override {
    program.AddUniformVariables({gsl::make_span(attr, 2)});
    return Status::OK();
  }

 protected:
  float attr[2];
};

WEBGPU_ELEMENTWISE_KERNEL(HardSigmoid, 6, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(HardSwish, "hard_swish_v(a)", HardSwishImpl, ShaderUsage::UseElementTypeAlias)
WEBGPU_ELEMENTWISE_KERNEL(HardSwish, 14, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Sin, "sin(a)")
WEBGPU_ELEMENTWISE_KERNEL(Sin, 7, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Cos, "cos(a)")
WEBGPU_ELEMENTWISE_KERNEL(Cos, 7, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Tan, "tan(a)")
WEBGPU_ELEMENTWISE_KERNEL(Tan, 7, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Asin, "asin(a)")
WEBGPU_ELEMENTWISE_KERNEL(Asin, 7, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Acos, "acos(a)")
WEBGPU_ELEMENTWISE_KERNEL(Acos, 7, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Atan, "atan(a)")
WEBGPU_ELEMENTWISE_KERNEL(Atan, 7, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Sinh, "sinh(a)")
WEBGPU_ELEMENTWISE_KERNEL(Sinh, 9, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Cosh, "cosh(a)")
WEBGPU_ELEMENTWISE_KERNEL(Cosh, 9, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Tanh, "tanh_v(a)", TanhImpl, ShaderUsage::UseValueTypeAlias)
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Tanh, 6, 12, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Tanh, 13, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Asinh, "asinh(a)")
WEBGPU_ELEMENTWISE_KERNEL(Asinh, 9, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Acosh, "acosh(a)")
WEBGPU_ELEMENTWISE_KERNEL(Acosh, 9, WebGpuSupportedFloatTypes())

#if __APPLE__
// Metal returns 0 for values >= 1.0.
// Need custom impl to return +inf for 1.0 (by dividing 1 by 0), and NaN for > 1.0 (by dividing 0 by 0)
WEBGPU_ELEMENTWISE_IMPL(Atanh,
                        "select("
                        " select(x_value_t(1.0), x_value_t(0.0), a > x_value_t(1.0)) / x_value_t(0.0),"
                        " atanh(a),"
                        " a < x_value_t(1.0))",
                        "",
                        ShaderUsage::UseValueTypeAlias)
#else
WEBGPU_ELEMENTWISE_IMPL(Atanh, "atanh(a)")
#endif
WEBGPU_ELEMENTWISE_KERNEL(Atanh, 9, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Not, "!a")
WEBGPU_ELEMENTWISE_BOOLEAN_KERNEL(Not, 1)

// No longer support Clip < opset 11 (where min and max are attributes)
//
// Use template class for "Clip" because the implementation is significantly different between float16 and float32
template <typename T>
class Clip final : public UnaryElementwise {
 public:
  Clip(const OpKernelInfo& info)
      : UnaryElementwise{info,
                         "Clip",
                         std::is_same_v<T, MLFloat16> ? ClipF16Impl : ClipImpl,
                         "", ShaderUsage::UseElementTypeAlias} {}

  Status ConfigureProgram(const ComputeContext& context, UnaryElementwiseProgram& program) const override {
    const auto* clip_min_tensor = context.Input<Tensor>(1);
    const auto* clip_max_tensor = context.Input<Tensor>(2);

    const T attr[] = {clip_min_tensor ? clip_min_tensor->Data<T>()[0]
                                      : std::numeric_limits<T>::lowest(),
                      clip_max_tensor ? clip_max_tensor->Data<T>()[0]
                                      : std::numeric_limits<T>::max()};
    if constexpr (std::is_same_v<T, MLFloat16>) {
      // F16: pack the two f16 values into a single f32 uniform slot; the shader unpacks with
      // bitcast<vec2<f16>>.
      float encoded_value;
      static_assert(sizeof(encoded_value) == 2 * sizeof(MLFloat16));
      std::memcpy(&encoded_value, attr, sizeof(encoded_value));
      program.AddUniformVariable({encoded_value});
    } else if constexpr (std::is_same_v<T, float>) {
      // f32: stored as-is.
      program.AddUniformVariable({gsl::make_span(attr, 2)});
    } else {
      // i32 / u32: the "attr" uniform is declared f32 and the WebGPU EP validates that the supplied
      // uniform value's data type matches the declaration. Reinterpret the integer bits as f32 so the
      // types match; the shader recovers the integer values with bitcast<x_element_t>(uniforms.attr[i]).
      static_assert(sizeof(T) == sizeof(float), "integer Clip attr must be 4 bytes");
      float encoded[2];
      std::memcpy(encoded, attr, sizeof(encoded));
      program.AddUniformVariable({gsl::make_span(encoded, 2)});
    }
    return Status::OK();
  }

  // uniforms.attr is a f32 value. It is encoded as a float for 2 f16 values.
  // bitcast<vec2<f16>>(uniforms.attr)[0] is clip_min, bitcast<vec2<f16>>(uniforms.attr)[1] is clip_max
  constexpr static const char ClipF16Impl[] = "clamp(a, vec4<f16>(bitcast<vec2<f16>>(uniforms.attr)[0]), vec4<f16>(bitcast<vec2<f16>>(uniforms.attr)[1]))";

  // the size of element of uniforms.attr should be the same as x_element_t. use bitcast to convert between them
  // uniforms.attr[0] is clip_min, uniforms.attr[1] is clip_max
  constexpr static const char ClipImpl[] = "clamp(a, vec4<x_element_t>(bitcast<x_element_t>(uniforms.attr[0])), vec4<x_element_t>(bitcast<x_element_t>(uniforms.attr[1])))";
};

// Clip for int64 tensors. WebGPU has no native 64-bit integer type; the EP stores int64 as
// vec2<u32> but, by convention, reads/writes it as the truncated low 32 bits interpreted as i32
// (see shader_variable.cc GetByOffset/SetByOffset for Int64). Because Clip is monotonic, clamping
// the truncated value and sign-extending on write is consistent with that existing int64 handling
// and correct for the index/position ranges that use int64 Clip in practice. The 4-byte-only
// templated Clip above (sizeof(T)==sizeof(float) static_assert) cannot cover int64, so it gets a
// dedicated one-element-per-invocation program here.
class ClipInt64Program final : public Program<ClipInt64Program> {
 public:
  ClipInt64Program() : Program{"ClipInt64"} {}

  Status GenerateShaderCode(ShaderHelper& sh) const override {
    const auto& input = sh.AddInput("x", ShaderUsage::UseUniform);
    const auto& output = sh.AddOutput("y", ShaderUsage::UseUniform);
    sh.MainFunctionBody() << sh.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size")
                          << "  let a = " << input.GetByOffset("global_idx") << ";\n"
                          << "  let clamped = min(max(a, uniforms.clip_min), uniforms.clip_max);\n  "
                          << output.SetByOffset("global_idx", "clamped");
    return Status::OK();
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"vec_size", ProgramUniformVariableDataType::Uint32},
                                          {"clip_min", ProgramUniformVariableDataType::Int32},
                                          {"clip_max", ProgramUniformVariableDataType::Int32});
};

class ClipInt64 final : public WebGpuKernel {
 public:
  ClipInt64(const OpKernelInfo& info) : WebGpuKernel{info} {}

  Status ComputeInternal(ComputeContext& context) const override {
    const auto* input_tensor = context.Input(0);
    auto* output_tensor = context.Output(0, input_tensor->Shape());
    int64_t size = input_tensor->Shape().Size();
    if (size == 0) {
      return Status::OK();
    }

    // min/max arrive as CPU scalar inputs (see InputMemoryType below). Saturate them into the i32
    // range that the shader operates on: values live in the low 32 bits, so an out-of-i32-range
    // bound simply means "no clamp on that side".
    constexpr int64_t kI32Min = std::numeric_limits<int32_t>::lowest();
    constexpr int64_t kI32Max = std::numeric_limits<int32_t>::max();
    const auto* clip_min_tensor = context.Input<Tensor>(1);
    const auto* clip_max_tensor = context.Input<Tensor>(2);
    auto saturate_to_i32 = [](int64_t v) -> int32_t {
      return static_cast<int32_t>(v < kI32Min ? kI32Min : (v > kI32Max ? kI32Max : v));
    };
    int32_t clip_min = clip_min_tensor ? saturate_to_i32(clip_min_tensor->Data<int64_t>()[0])
                                       : static_cast<int32_t>(kI32Min);
    int32_t clip_max = clip_max_tensor ? saturate_to_i32(clip_max_tensor->Data<int64_t>()[0])
                                       : static_cast<int32_t>(kI32Max);

    // The shader carries the element count in a u32 uniform, so validate the range explicitly and
    // return INVALID_ARGUMENT rather than letting narrow<uint32_t> hard-terminate the process in
    // ORT_NO_EXCEPTIONS builds (as the WebGPU EP is typically built).
    if (size > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ClipInt64 input has ", size,
                             " elements, which exceeds the WebGPU supported maximum of ",
                             std::numeric_limits<uint32_t>::max(), ".");
    }
    uint32_t data_size = static_cast<uint32_t>(size);
    ClipInt64Program program{};
    // Uniform values are positional: this order must match the WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES
    // declaration on ClipInt64Program (vec_size, clip_min, clip_max).
    program.AddInput({input_tensor, ProgramTensorMetadataDependency::Type, {size}, 1})
        .AddOutput({output_tensor, ProgramTensorMetadataDependency::None, {size}, 1})
        .SetDispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({{data_size}, {clip_min}, {clip_max}});
    return context.RunProgram(program);
  }
};

#define WEBGPU_CLIP_KERNEL(TYPE)                                                                        \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(Clip, kOnnxDomain, 11, 11, TYPE, kWebGpuExecutionProvider,    \
                                          KernelDefBuilder()                                            \
                                              .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()) \
                                              .InputMemoryType(OrtMemTypeCPU, 1)                        \
                                              .InputMemoryType(OrtMemTypeCPU, 2),                       \
                                          Clip<TYPE>)                                                   \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(Clip, kOnnxDomain, 12, 12, TYPE, kWebGpuExecutionProvider,    \
                                          KernelDefBuilder()                                            \
                                              .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()) \
                                              .InputMemoryType(OrtMemTypeCPU, 1)                        \
                                              .InputMemoryType(OrtMemTypeCPU, 2),                       \
                                          Clip<TYPE>)                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Clip, kOnnxDomain, 13, TYPE, kWebGpuExecutionProvider,                  \
                                KernelDefBuilder()                                                      \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())           \
                                    .InputMemoryType(OrtMemTypeCPU, 1)                                  \
                                    .InputMemoryType(OrtMemTypeCPU, 2),                                 \
                                Clip<TYPE>);

// Same as WEBGPU_CLIP_KERNEL but without the 11-11 registration: integer types are only valid Clip
// element types from opset 12 on (the opset-11 Clip schema constrains T to float types), so an
// 11-11 integer registration would be dead -- no valid opset-11 model can have an integer Clip.
#define WEBGPU_CLIP_KERNEL_FROM_12(TYPE)                                                                \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(Clip, kOnnxDomain, 12, 12, TYPE, kWebGpuExecutionProvider,    \
                                          KernelDefBuilder()                                            \
                                              .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>()) \
                                              .InputMemoryType(OrtMemTypeCPU, 1)                        \
                                              .InputMemoryType(OrtMemTypeCPU, 2),                       \
                                          Clip<TYPE>)                                                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Clip, kOnnxDomain, 13, TYPE, kWebGpuExecutionProvider,                  \
                                KernelDefBuilder()                                                      \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<TYPE>())           \
                                    .InputMemoryType(OrtMemTypeCPU, 1)                                  \
                                    .InputMemoryType(OrtMemTypeCPU, 2),                                 \
                                Clip<TYPE>);

// int64 Clip uses the dedicated ClipInt64 kernel (defined above), not the 4-byte templated Clip.
// int64 (like all integer types) is only a valid Clip element type from opset 12 on -- the opset-11
// Clip schema constrains T to float types only -- so there is no 11-11 registration here.
#define WEBGPU_CLIP_INT64_KERNEL()                                                                         \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(Clip, kOnnxDomain, 12, 12, int64_t, kWebGpuExecutionProvider,    \
                                          KernelDefBuilder()                                               \
                                              .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()) \
                                              .InputMemoryType(OrtMemTypeCPU, 1)                           \
                                              .InputMemoryType(OrtMemTypeCPU, 2),                          \
                                          ClipInt64)                                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Clip, kOnnxDomain, 13, int64_t, kWebGpuExecutionProvider,                  \
                                KernelDefBuilder()                                                         \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>())           \
                                    .InputMemoryType(OrtMemTypeCPU, 1)                                     \
                                    .InputMemoryType(OrtMemTypeCPU, 2),                                    \
                                ClipInt64);

WEBGPU_CLIP_KERNEL(float)
WEBGPU_CLIP_KERNEL(MLFloat16)
// Integer Clip is used by shape/index/mask subgraphs (e.g. an index/position path). The 4-byte
// templated Clip covers int32/uint32. Registered from opset 12 only (see WEBGPU_CLIP_KERNEL_FROM_12).
WEBGPU_CLIP_KERNEL_FROM_12(int32_t)
WEBGPU_CLIP_KERNEL_FROM_12(uint32_t)
// int64 Clip (e.g. an int64 Clip on an index/position path feeding ArgMax) is handled by the
// dedicated ClipInt64 kernel above; the 4-byte templated Clip cannot cover int64.
WEBGPU_CLIP_INT64_KERNEL()

//
// activation
//

#define WEBGPU_LU_IMPL(OP_TYPE, ...)                                               \
  class OP_TYPE final : public LinearUnit {                                        \
   public:                                                                         \
    OP_TYPE(const OpKernelInfo& info) : LinearUnit{info, #OP_TYPE, __VA_ARGS__} {} \
  };

WEBGPU_LU_IMPL(Elu, "elu_v(a)", EluImpl, 1.0)
WEBGPU_ELEMENTWISE_KERNEL(Elu, 6, WebGpuSupportedFloatTypes())

Gelu::Gelu(const OpKernelInfo& info)
    : UnaryElementwise{info,
                       "Gelu",
                       info.GetAttrOrDefault<std::string>("approximate", "none") == "tanh" ? FastGeluExpr : GeluExpr,
                       info.GetAttrOrDefault<std::string>("approximate", "none") == "tanh" ? TanhImpl : ErfImpl,
                       ShaderUsage::UseValueTypeAlias} {
  cache_hint = info.GetAttrOrDefault<std::string>("approximate", "none");
}

QuickGelu::QuickGelu(const OpKernelInfo& info)
    : LinearUnit{info, "QuickGelu", "quick_gelu_v(a)", QuickGeluImpl, 1.702f} {}

WEBGPU_ELEMENTWISE_KERNEL(Gelu, 20, WebGpuSupportedFloatTypes())

WEBGPU_ELEMENTWISE_IMPL(Relu, "select(x_value_t(0), a, a > x_value_t(0))", "", ShaderUsage::UseValueTypeAlias)
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Relu, 6, 12, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Relu, 13, 13, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Relu, 14, WebGpuSupportedFloatTypes())

WEBGPU_LU_IMPL(LeakyRelu, "select(x_element_t(uniforms.attr) * a, a, a >= vec4<x_element_t>(0))", "", 0.01f)
WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(LeakyRelu, 6, 15, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(LeakyRelu, 16, WebGpuSupportedFloatTypes())

WEBGPU_LU_IMPL(ThresholdedRelu, "select(vec4<x_element_t>(0), a, a > vec4<x_element_t>(uniforms.attr))", "", 1.0f)
WEBGPU_ELEMENTWISE_KERNEL(ThresholdedRelu, 10, WebGpuSupportedFloatTypes())

// For large a, softplus(a) = log(1 + exp(a)) ≈ a. Use a threshold to return a directly,
// avoiding unnecessary exp/log computation and potential overflow.
// PyTorch uses threshold=20 for float32. For float16, exp overflows at ~11.09 so use 11.
class Softplus final : public UnaryElementwise {
 public:
  Softplus(const OpKernelInfo& info)
      : UnaryElementwise{info, "Softplus",
                         "select("
                         "select(log(1.0 + exp(a)), a + log(1.0 + exp(-a)), a > x_value_t(0)),"
                         "a,"
                         "a > x_value_t(x_element_t(uniforms.attr))"
                         ")",
                         "",
                         ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias} {}

  Status ConfigureProgram(const ComputeContext& context, UnaryElementwiseProgram& program) const override {
    const auto* input_tensor = context.Input<Tensor>(0);
    float threshold = input_tensor->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? 11.0f : 20.0f;
    program.AddUniformVariables({threshold});
    return Status::OK();
  }
};

WEBGPU_ELEMENTWISE_VERSIONED_KERNEL(Softplus, 1, 21, WebGpuSupportedFloatTypes())
WEBGPU_ELEMENTWISE_KERNEL(Softplus, 22, WebGpuSupportedFloatTypes())

}  // namespace webgpu
}  // namespace onnxruntime
