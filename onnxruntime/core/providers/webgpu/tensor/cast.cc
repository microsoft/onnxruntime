// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <string>
#include <vector>

#include "core/providers/webgpu/tensor/cast.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

Status Cast::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  auto* output_tensor = context.Output(0, input_tensor->Shape());
  int64_t size = input_tensor->Shape().Size();
  if (size == 0) {
    return Status::OK();
  }
  bool is_from_int64 = input_tensor->DataType() == DataTypeImpl::GetType<int64_t>();
  bool is_from_float = input_tensor->DataType() == DataTypeImpl::GetType<float>() ||
                       input_tensor->DataType() == DataTypeImpl::GetType<MLFloat16>();
  bool is_from_unsigned = input_tensor->DataType() == DataTypeImpl::GetType<uint32_t>() ||
                          input_tensor->DataType() == DataTypeImpl::GetType<bool>();
  const int in_components = is_from_int64 ? 1 : 4;
  const int out_components = to_ == ONNX_NAMESPACE::TensorProto_DataType_INT64 ? 1 : 4;
  uint32_t vec_size = onnxruntime::narrow<uint32_t>((size + 3) / 4);
  uint32_t in_vec_size = onnxruntime::narrow<uint32_t>(in_components == 1 ? size : vec_size);
  uint32_t out_vec_size = onnxruntime::narrow<uint32_t>(out_components == 1 ? size : vec_size);

  CastProgram program{to_, is_from_int64, is_from_float, is_from_unsigned};
  program
      .AddInput({input_tensor, ProgramTensorMetadataDependency::Type, {in_vec_size}, in_components})
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::None, {out_vec_size}, out_components})
      .SetDispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({
          {static_cast<uint32_t>(vec_size)},
          {static_cast<uint32_t>(size)},
      })
      .CacheHint(std::to_string(to_));
  return context.RunProgram(program);
}

Status CastProgram::GenerateShaderCode(ShaderHelper& sh) const {
  const auto& input = sh.AddInput("x", ShaderUsage::UseUniform);
  const auto& output = sh.AddOutput("y", ShaderUsage::UseUniform);
  std::string expression;
  switch (to_) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      expression = "vec4<f16>(a)";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      expression = "vec4<f32>(a)";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      expression = "vec4<i32>(a)";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      expression = "vec4<u32>(a)";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      expression = "vec4<bool>(a)";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      expression = "int32(a)";
      break;
    default:
      ORT_NOT_IMPLEMENTED("Cast to type ", to_, " is not supported.");
  }

  // float32 -> int64 via IEEE 754 bit decomposition:
  //   - Finite values: truncated toward zero (matches C++ static_cast<int64_t>).
  //   - Zero/subnormals: 0.
  //   - Out-of-range, +/-Inf, NaN: saturate by sign to INT64_MAX / INT64_MIN.
  // Saturation is spec-compliant (ONNX leaves these undefined) but differs from x86 static_cast,
  // so don't pin a specific NaN->int64 result in cross-EP tests.
  if (to_ == ONNX_NAMESPACE::TensorProto_DataType_INT64 && is_from_float_) {
    sh.AdditionalImplementation() << "fn float_to_int64(f: f32) -> vec2<u32> {\n"
                                     "  let bits = bitcast<u32>(f);\n"
                                     "  let sign = (bits >> 31u) & 1u;\n"
                                     "  let biased_exp = (bits >> 23u) & 0xFFu;\n"
                                     "  let mantissa = bits & 0x7FFFFFu;\n"
                                     "  if (biased_exp == 0u) {\n"
                                     "    return vec2<u32>(0u, 0u);\n"
                                     "  }\n"
                                     "  if (biased_exp == 255u) {\n"
                                     "    return select(vec2<u32>(0xFFFFFFFFu, 0x7FFFFFFFu),\n"
                                     "                  vec2<u32>(0u, 0x80000000u), sign == 1u);\n"
                                     "  }\n"
                                     "  let sig = 0x800000u | mantissa;\n"
                                     "  let exp = i32(biased_exp) - 150;\n"
                                     "  var low: u32 = 0u;\n"
                                     "  var high: u32 = 0u;\n"
                                     "  if (exp >= 40) {\n"
                                     "    return select(vec2<u32>(0xFFFFFFFFu, 0x7FFFFFFFu),\n"
                                     "                  vec2<u32>(0u, 0x80000000u), sign == 1u);\n"
                                     "  } else if (exp >= 32) {\n"
                                     "    high = sig << u32(exp - 32);\n"
                                     "  } else if (exp > 0) {\n"
                                     "    low = sig << u32(exp);\n"
                                     "    high = sig >> u32(32 - exp);\n"
                                     "  } else if (exp == 0) {\n"
                                     "    low = sig;\n"
                                     "  } else if (exp > -24) {\n"
                                     "    low = sig >> u32(-exp);\n"
                                     "  }\n"
                                     "  if (sign == 1u) {\n"
                                     "    let carry = select(0u, 1u, low == 0u);\n"
                                     "    low = ~low + 1u;\n"
                                     "    high = ~high + carry;\n"
                                     "  }\n"
                                     "  return vec2<u32>(low, high);\n"
                                     "}\n";
  }

  sh.MainFunctionBody() << sh.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size");

  if (is_from_int64_) {
    // int64 -> any (including int64)
    // Note: int64 inputs are not enabled by default (requires enable_int64).
    // This path handles the downcast to 32-bit types.
    // Load lanes 1-3 conditionally to avoid out-of-bounds reads when size % 4 != 0.
    // Lane 0 is always valid due to the workgroup size guard.
    sh.MainFunctionBody() << "  let base = global_idx * 4u;\n"
                          << "  let a0 = " << input.GetByOffset("base") << ";\n"
                          << "  var a1: i32 = 0;\n"
                          << "  var a2: i32 = 0;\n"
                          << "  var a3: i32 = 0;\n";
    for (size_t i = 1; i < 4; ++i) {
      sh.MainFunctionBody() << "  if (base + " << i << "u < uniforms.output_size) { a" << i
                            << " = " << input.GetByOffset(MakeStringWithClassicLocale("base + ", i, "u")) << "; }\n";
    }
    sh.MainFunctionBody() << "  let a = vec4<i32>(a0, a1, a2, a3);\n";
    if (to_ == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      // int64 -> int64
      constexpr std::array<char, 4> kLanes{'x', 'y', 'z', 'w'};
      sh.MainFunctionBody() << output.SetByOffset("base", "a.x");
      for (size_t i = 1; i < 4; ++i) {
        sh.MainFunctionBody() << "  if (base + " << i << "u < uniforms.output_size) { "
                              << output.SetByOffset(MakeStringWithClassicLocale("base + ", i, "u"),
                                                    MakeStringWithClassicLocale("a.", kLanes[i]))
                              << " }\n";
      }
    } else {
      sh.MainFunctionBody() << output.SetByOffset("global_idx", expression);
    }
  } else if (to_ == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    // cast to int64 (non-int64 inputs only)
    std::array<std::string, 4> values;
    constexpr std::array<char, 4> kLanes{'x', 'y', 'z', 'w'};
    sh.MainFunctionBody() << "  let a = " << input.GetByOffset("global_idx") << ";\n"
                          << "  let base = global_idx * 4u;\n";
    for (size_t i = 0; i < 4; ++i) {
      if (is_from_float_) {
        // float32/float16 -> int64: IEEE 754 bit decomposition. float16 reuses the
        // float32 helper since every f16 value is exactly representable as f32.
        values[i] = MakeStringWithClassicLocale("float_to_int64(f32(a.", kLanes[i], "))");
      } else if (is_from_unsigned_) {
        // uint32/bool -> int64: zero-extend.
        values[i] = MakeStringWithClassicLocale("vec2<u32>(u32(a.", kLanes[i], "), 0u)");
      } else {
        // int32 -> int64: sign-extend.
        values[i] = MakeStringWithClassicLocale(
            "vec2<u32>(u32(a.", kLanes[i], "), select(0u, 0xFFFFFFFFu, i32(a.", kLanes[i], ") < 0))");
      }
    }
    // Use use_storage_type=true to write vec2<u32> directly.
    sh.MainFunctionBody() << output.SetByOffset("base", values[0], /*use_storage_type=*/true);
    for (size_t i = 1; i < 4; ++i) {
      sh.MainFunctionBody() << "  if (base + " << i << "u < uniforms.output_size) { "
                            << output.SetByOffset(MakeStringWithClassicLocale("base + ", i, "u"), values[i], /*use_storage_type=*/true)
                            << " }\n";
    }
  } else {
    // generic cast (no int64 involved).
    sh.MainFunctionBody() << "  let a = " << input.GetByOffset("global_idx") << ";\n";
    sh.MainFunctionBody() << output.SetByOffset("global_idx", expression);
  }

  return Status::OK();
}

template <int StartVersion, int EndVersion>
KernelCreateInfo CreateCastKernelInfo(bool enable_int64) {
  // Casting to int64 is always supported. Casting *from* int64 (int64 in T1/input) stays guarded by enable_int64.
  const auto& t1_constraints = GetOpTypeConstraints(/*enable_int64=*/enable_int64, /*enable_bool=*/true);
  const auto& t2_constraints = GetOpTypeConstraints(/*enable_int64=*/true, /*enable_bool=*/true);

  KernelCreatePtrFn kernel_create_fn = [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
    out = std::make_unique<Cast>(info);
    return Status::OK();
  };

  if constexpr (StartVersion == EndVersion) {
    return {
        KernelDefBuilder()
            .SetName("Cast")
            .SetDomain(kOnnxDomain)
            .SinceVersion(StartVersion)
            .Provider(kWebGpuExecutionProvider)
            .TypeConstraint("T1", t1_constraints)
            .TypeConstraint("T2", t2_constraints)
            .Build(),
        kernel_create_fn};
  } else {
    return {
        KernelDefBuilder()
            .SetName("Cast")
            .SetDomain(kOnnxDomain)
            .SinceVersion(StartVersion, EndVersion)
            .Provider(kWebGpuExecutionProvider)
            .TypeConstraint("T1", t1_constraints)
            .TypeConstraint("T2", t2_constraints)
            .Build(),
        kernel_create_fn};
  }
}

// Explicit template instantiations
template KernelCreateInfo CreateCastKernelInfo<6, 8>(bool);
template KernelCreateInfo CreateCastKernelInfo<9, 12>(bool);
template KernelCreateInfo CreateCastKernelInfo<13, 18>(bool);
template KernelCreateInfo CreateCastKernelInfo<19, 20>(bool);
template KernelCreateInfo CreateCastKernelInfo<21, 22>(bool);
template KernelCreateInfo CreateCastKernelInfo<23, 23>(bool);
template KernelCreateInfo CreateCastKernelInfo<24>(bool);

}  // namespace webgpu
}  // namespace onnxruntime
