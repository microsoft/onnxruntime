// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <utility>
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
  uint32_t vec_size = gsl::narrow<uint32_t>((size + 3) / 4);
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
      // F16: stores span<f16, 2> as a single float
      float encoded_value = *reinterpret_cast<const float*>(attr);
      program.AddUniformVariable({encoded_value});
    } else {
      static_assert(sizeof(T) == sizeof(float), "T must be f32, i32 or u32");
      // stores span<f32, 2> as-is
      program.AddUniformVariable({gsl::make_span(attr, 2)});
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
WEBGPU_CLIP_KERNEL(float)
WEBGPU_CLIP_KERNEL(MLFloat16)

//
// activation
//

class LinearUnit : public UnaryElementwise {
 public:
  LinearUnit(const OpKernelInfo& info,
             const std::string& kernel_name,
             const std::string& expression,
             const std::string& additional_impl,
             float default_alpha)
      : UnaryElementwise{info, kernel_name, expression, additional_impl, ShaderUsage::UseElementTypeAlias} {
    info.GetAttrOrDefault("alpha", &alpha_, default_alpha);
  }

  Status ConfigureProgram(const ComputeContext& /*context*/, UnaryElementwiseProgram& program) const override {
    program.AddUniformVariables({alpha_});
    return Status::OK();
  }

 protected:
  float alpha_;
};

#define WEBGPU_LU_IMPL(OP_TYPE, ...)                                               \
  class OP_TYPE final : public LinearUnit {                                        \
   public:                                                                         \
    OP_TYPE(const OpKernelInfo& info) : LinearUnit{info, #OP_TYPE, __VA_ARGS__} {} \
  };

WEBGPU_LU_IMPL(Elu, "elu_v(a)", EluImpl, 1.0)
WEBGPU_ELEMENTWISE_KERNEL(Elu, 6, WebGpuSupportedFloatTypes())

class Gelu : public UnaryElementwise {
 public:
  Gelu(const OpKernelInfo& info)
      : UnaryElementwise{info,
                         "Gelu",
                         info.GetAttrOrDefault<std::string>("approximate", "none") == "tanh" ? FastGeluExpr : GeluExpr,
                         info.GetAttrOrDefault<std::string>("approximate", "none") == "tanh" ? TanhImpl : ErfImpl,
                         ShaderUsage::UseValueTypeAlias} {
    cache_hint = info.GetAttrOrDefault<std::string>("approximate", "none");
  }
};

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

}  // namespace webgpu
}  // namespace onnxruntime
