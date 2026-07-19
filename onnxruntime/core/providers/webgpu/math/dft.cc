// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/dft.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/providers/common.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {

// ONNX DFT (opset 17-20): a 1-D transform along `axis`, with the innermost dimension holding the
// real/imaginary parts (1 for real input, 2 for complex). Forward is unnormalized, inverse is scaled
// by 1/N -- matching the CPU kernel core/providers/cpu/signal/dft.cc. 5-smooth lengths use a
// shared-memory mixed-radix (2/3/4/5) Stockham FFT (one transform per workgroup); other lengths fall
// back to a direct O(N^2) DFT (the CPU kernel uses Bluestein chirp-z there, which would restore
// O(N log N) if the fallback ever becomes hot).

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    DFT,
    kOnnxDomain,
    17, 19,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int32_t>(),
                               DataTypeImpl::GetTensorType<int64_t>()})
        .InputMemoryType(OrtMemTypeCPU, 1),
    DFT);

ONNX_OPERATOR_KERNEL_EX(
    DFT,
    kOnnxDomain,
    20,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", WebGpuSupportedFloatTypes())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int32_t>(),
                               DataTypeImpl::GetTensorType<int64_t>()})
        .InputMemoryType(OrtMemTypeCPU, 1)
        .InputMemoryType(OrtMemTypeCPU, 2),
    DFT);

static constexpr uint32_t kDftWorkgroupSize = 256;
static constexpr uint32_t kDftMaxSharedMemoryLength = 512;
static constexpr double kTwoPi = 6.283185307179586476925286766559;

static bool FactorizeToRadices(uint32_t length, std::vector<uint32_t>& radices) {
  uint32_t remaining = length;
  for (uint32_t radix : {4u, 2u, 3u, 5u}) {
    while (remaining % radix == 0) {
      radices.push_back(radix);
      remaining /= radix;
    }
  }
  return remaining == 1;
}

static std::string WgslFloat(double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(9) << value;
  return oss.str();
}

// One radix-`radix` Stockham stage: reads the `sub_transform`-sized partial transforms from
// `read_offset` and writes the combined transforms to the other half of the ping-pong buffer,
// with twiddles baked in.
static void EmitFftStage(OStringStream& os, uint32_t radix, uint32_t sub_transform, uint32_t length,
                         uint32_t read_offset, int sign) {
  const uint32_t butterflies = length / radix;
  const uint32_t write_offset = kDftMaxSharedMemoryLength - read_offset;
  auto out = [&](uint32_t k) {
    return "smem[" + std::to_string(write_offset) + "u + base + " + std::to_string(k * sub_transform) + "u]";
  };
  os << "  for (var t = local_idx; t < " << butterflies << "u; t += " << kDftWorkgroupSize << "u) {\n"
     << "    let twiddle_index = t % " << sub_transform << "u;\n"
     << "    let angle_unit = f32(twiddle_index);\n"
     << "    var leg: array<vec2<f32>, 5>;\n";
  for (uint32_t j = 0; j < radix; ++j) {
    const std::string source = std::to_string(read_offset) + "u + t + " + std::to_string(j * butterflies) + "u";
    if (j == 0) {
      os << "    leg[0] = smem[" << source << "];\n";
    } else {
      const double twiddle = (sign * kTwoPi * j) / (radix * sub_transform);
      os << "    { let a = " << WgslFloat(twiddle) << " * angle_unit; leg[" << j
         << "] = cmul(smem[" << source << "], vec2<f32>(cos(a), sin(a))); }\n";
    }
  }
  os << "    let base = (t / " << sub_transform << "u) * " << sub_transform * radix << "u + twiddle_index;\n";
  if (radix == 2) {
    os << "    " << out(0) << " = leg[0] + leg[1];\n"
       << "    " << out(1) << " = leg[0] - leg[1];\n";
  } else if (radix == 4) {
    const std::string rotate = sign < 0 ? "vec2<f32>(odd_diff.y, -odd_diff.x)" : "vec2<f32>(-odd_diff.y, odd_diff.x)";
    os << "    let even_sum = leg[0] + leg[2]; let even_diff = leg[0] - leg[2];\n"
       << "    let odd_sum = leg[1] + leg[3]; let odd_diff = leg[1] - leg[3];\n"
       << "    let odd_rot = " << rotate << ";\n"
       << "    " << out(0) << " = even_sum + odd_sum;\n"
       << "    " << out(1) << " = even_diff + odd_rot;\n"
       << "    " << out(2) << " = even_sum - odd_sum;\n"
       << "    " << out(3) << " = even_diff - odd_rot;\n";
  } else {
    for (uint32_t k = 0; k < radix; ++k) {
      os << "    " << out(k) << " = leg[0]";
      for (uint32_t j = 1; j < radix; ++j) {
        const double angle = (sign * kTwoPi * (j * k)) / radix;
        const std::string c = WgslFloat(std::cos(angle));
        const std::string s = WgslFloat(std::sin(angle));
        os << " + vec2<f32>(leg[" << j << "].x*" << c << " - leg[" << j << "].y*" << s
           << ", leg[" << j << "].x*" << s << " + leg[" << j << "].y*" << c << ")";
      }
      os << ";\n";
    }
  }
  os << "  }\n"
     << "  workgroupBarrier();\n";
}

Status DFTProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("x", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  const int sign = is_inverse_ ? 1 : -1;
  const double scale = is_inverse_ ? 1.0 / length_ : 1.0;

  std::vector<uint32_t> radices;
  ORT_RETURN_IF_NOT(FactorizeToRadices(length_, radices), "DFT length ", length_, " is not 5-smooth.");

  shader.AdditionalImplementation()
      << "var<workgroup> smem: array<vec2<f32>, " << 2 * kDftMaxSharedMemoryLength << ">;\n"
      << "fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {\n"
      << "  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);\n"
      << "}\n";

  auto read_sample = [&](const std::string& index) {
    const std::string offset = "in_base + (" + index + ") * uniforms.inner * " + std::to_string(input_components_) + "u";
    const std::string real = "f32(" + input.GetByOffset(offset) + ")";
    const std::string imag = input_components_ == 2 ? "f32(" + input.GetByOffset(offset + " + 1u") + ")" : "0.0";
    return "vec2<f32>(" + real + ", " + imag + ")";
  };

  shader.MainFunctionBody()
      << "let row = workgroup_idx;\n"
      << "if (row >= uniforms.batch) { return; }\n"
      << "let outer = row / uniforms.inner;\n"
      << "let within = row % uniforms.inner;\n"
      << "let in_base = (outer * uniforms.signal_length * uniforms.inner + within) * " << input_components_ << "u;\n"
      << "let out_base = (outer * uniforms.output_length * uniforms.inner + within) * " << output_components_ << "u;\n";

  if (is_inverse_ && is_onesided_) {
    // For IRFFT the spectrum is the Hermitian extension of the half-spectrum input.
    const std::string conjugate_end = length_ % 2 == 0 ? "uniforms.signal_length - 1u" : "uniforms.signal_length";
    shader.MainFunctionBody()
        << "for (var i = local_idx; i < uniforms.signal_length; i += " << kDftWorkgroupSize << "u) {\n"
        << "  smem[i] = " << read_sample("i") << ";\n"
        << "}\n"
        << "workgroupBarrier();\n"
        << "for (var k = local_idx + 1u; k < " << conjugate_end << "; k += " << kDftWorkgroupSize << "u) {\n"
        << "  let h = smem[k];\n"
        << "  smem[" << length_ << "u - k] = vec2<f32>(h.x, -h.y);\n"
        << "}\n"
        << "workgroupBarrier();\n";
  } else {
    shader.MainFunctionBody()
        << "let load_count = min(uniforms.signal_length, " << length_ << "u);\n"
        << "for (var i = local_idx; i < " << length_ << "u; i += " << kDftWorkgroupSize << "u) {\n"
        << "  if (i < load_count) { smem[i] = " << read_sample("i") << "; } else { smem[i] = vec2<f32>(0.0); }\n"
        << "}\n"
        << "workgroupBarrier();\n";
  }

  uint32_t sub_transform = 1;
  uint32_t read_offset = 0;
  for (uint32_t radix : radices) {
    EmitFftStage(shader.MainFunctionBody(), radix, sub_transform, length_, read_offset, sign);
    sub_transform *= radix;
    read_offset = kDftMaxSharedMemoryLength - read_offset;
  }

  const std::string scaled = scale == 1.0
                                 ? "smem[" + std::to_string(read_offset) + "u + i]"
                                 : "smem[" + std::to_string(read_offset) + "u + i] * " + WgslFloat(scale);
  shader.MainFunctionBody()
      << "for (var i = local_idx; i < uniforms.output_length; i += " << kDftWorkgroupSize << "u) {\n"
      << "  let v = " << scaled << ";\n"
      << "  let off = out_base + i * uniforms.inner * " << output_components_ << "u;\n"
      << "  " << output.SetByOffset("off", "output_element_t(v.x)") << "\n";
  if (output_components_ == 2) {
    shader.MainFunctionBody()
        << "  " << output.SetByOffset("off + 1u", "output_element_t(v.y)") << "\n";
  }
  shader.MainFunctionBody() << "}\n";

  return Status::OK();
}

Status DFTDirectProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("x", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  const int sign = is_inverse_ ? 1 : -1;
  const double scale = is_inverse_ ? 1.0 / length_ : 1.0;

  auto read_sample = [&](const std::string& index) {
    const std::string offset = "in_base + (" + index + ") * uniforms.inner * " + std::to_string(input_components_) + "u";
    const std::string real = "f32(" + input.GetByOffset(offset) + ")";
    const std::string imag = input_components_ == 2 ? "f32(" + input.GetByOffset(offset + " + 1u") + ")" : "0.0";
    return "vec2<f32>(" + real + ", " + imag + ")";
  };

  shader.AdditionalImplementation()
      << "fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {\n"
      << "  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);\n"
      << "}\n";
  if (is_inverse_ && is_onesided_) {
    // For IRFFT the spectrum is the Hermitian extension of the half-spectrum input.
    shader.AdditionalImplementation()
        << "fn spectrum(in_base: u32, k: u32) -> vec2<f32> {\n"
        << "  if (k < uniforms.signal_length) { return " << read_sample("k") << "; }\n"
        << "  let h = " << read_sample(std::to_string(length_) + "u - k") << ";\n"
        << "  return vec2<f32>(h.x, -h.y);\n"
        << "}\n";
  } else {
    shader.AdditionalImplementation()
        << "fn spectrum(in_base: u32, n: u32) -> vec2<f32> {\n"
        << "  if (n < uniforms.signal_length) { return " << read_sample("n") << "; }\n"
        << "  return vec2<f32>(0.0, 0.0);\n"
        << "}\n";
  }

  const std::string scaled = scale == 1.0 ? "acc" : "acc * " + WgslFloat(scale);
  shader.MainFunctionBody()
      << "let row = workgroup_idx;\n"
      << "if (row >= uniforms.batch) { return; }\n"
      << "let outer = row / uniforms.inner;\n"
      << "let within = row % uniforms.inner;\n"
      << "let in_base = (outer * uniforms.signal_length * uniforms.inner + within) * " << input_components_ << "u;\n"
      << "let out_base = (outer * uniforms.output_length * uniforms.inner + within) * " << output_components_ << "u;\n"
      << "for (var k = local_idx; k < uniforms.output_length; k += " << kDftWorkgroupSize << "u) {\n"
      << "  var acc = vec2<f32>(0.0, 0.0);\n"
      // kn_mod tracks (k*n) mod length via addition, so the twiddle index never overflows u32 at large N.
      << "  var kn_mod = 0u;\n"
      << "  for (var n = 0u; n < " << length_ << "u; n++) {\n"
      << "    let angle = " << WgslFloat(sign * kTwoPi) << " * f32(kn_mod) / " << WgslFloat(static_cast<double>(length_)) << ";\n"
      << "    acc += cmul(spectrum(in_base, n), vec2<f32>(cos(angle), sin(angle)));\n"
      << "    kn_mod += k;\n"
      << "    if (kn_mod >= " << length_ << "u) { kn_mod -= " << length_ << "u; }\n"
      << "  }\n"
      << "  let v = " << scaled << ";\n"
      << "  let off = out_base + k * uniforms.inner * " << output_components_ << "u;\n"
      << "  " << output.SetByOffset("off", "output_element_t(v.x)") << "\n";
  if (output_components_ == 2) {
    shader.MainFunctionBody()
        << "  " << output.SetByOffset("off + 1u", "output_element_t(v.y)") << "\n";
  }
  shader.MainFunctionBody() << "}\n";

  return Status::OK();
}

static bool TryReadScalar(const Tensor* tensor, int64_t& value) {
  if (tensor == nullptr || tensor->Shape().Size() == 0) {
    return false;
  }
  if (tensor->DataType() == DataTypeImpl::GetType<int64_t>()) {
    value = tensor->Data<int64_t>()[0];
  } else {
    value = static_cast<int64_t>(tensor->Data<int32_t>()[0]);
  }
  return true;
}

Status DFT::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  const int64_t rank = static_cast<int64_t>(input_shape.NumDimensions());
  ORT_RETURN_IF(rank < 2, "DFT input must have at least 2 dimensions.");

  const int64_t input_components = input_shape[onnxruntime::narrow<size_t>(rank - 1)];
  ORT_RETURN_IF(input_components != 1 && input_components != 2,
                "DFT input's innermost dimension must be 1 (real) or 2 (complex).");

  int64_t axis = axis_;
  if (opset_ >= 20 && context.InputCount() > 2) {
    TryReadScalar(context.Input(2), axis);
  }
  axis = HandleNegativeAxis(axis, rank);
  ORT_RETURN_IF(axis == rank - 1,
                "DFT axis must refer to a signal dimension, not the innermost (real/imaginary) dimension.");
  ORT_RETURN_IF(is_inverse_ && is_onesided_ && input_components != 2,
                "Inverse one-sided DFT (IRFFT) requires complex-valued input (innermost dimension 2).");

  const int64_t signal_length = input_shape[onnxruntime::narrow<size_t>(axis)];
  int64_t length = is_inverse_ && is_onesided_ ? (signal_length - 1) * 2 : signal_length;
  if (context.InputCount() > 1) {
    TryReadScalar(context.Input(1), length);
  }
  ORT_RETURN_IF(length < 0 || length > std::numeric_limits<uint32_t>::max(), "Invalid DFT length: ", length);

  const int64_t output_components = is_inverse_ && is_onesided_ ? 1 : 2;
  const int64_t output_length = is_onesided_ && !is_inverse_ ? length / 2 + 1 : length;

  TensorShape output_shape = input_shape;
  output_shape[onnxruntime::narrow<size_t>(axis)] = output_length;
  output_shape[onnxruntime::narrow<size_t>(rank - 1)] = output_components;
  auto* output_tensor = context.Output(0, output_shape);
  if (output_shape.Size() == 0) {
    return Status::OK();
  }

  // Transforms packed between the axis and the innermost (real/imaginary) dimension.
  int64_t inner = 1;
  for (int64_t d = axis + 1; d < rank - 1; ++d) {
    inner *= input_shape[onnxruntime::narrow<size_t>(d)];
  }
  int64_t batch = 1;
  for (int64_t d = 0; d < rank - 1; ++d) {
    if (d != axis) {
      batch *= input_shape[onnxruntime::narrow<size_t>(d)];
    }
  }

  const uint32_t transform_length = static_cast<uint32_t>(length);
  std::vector<uint32_t> radices;
  const bool use_shared_memory_fft =
      transform_length <= kDftMaxSharedMemoryLength && FactorizeToRadices(transform_length, radices);

  if (use_shared_memory_fft) {
    DFTProgram program{transform_length, static_cast<uint32_t>(input_components),
                       static_cast<uint32_t>(output_components), is_inverse_, is_onesided_};
    program
        .AddInputs({{input_tensor, ProgramTensorMetadataDependency::Type}})
        .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::Type}})
        .CacheHint(std::to_string(transform_length), std::to_string(input_components),
                   std::to_string(output_components), std::to_string(is_inverse_), std::to_string(is_onesided_))
        .SetWorkgroupSize(kDftWorkgroupSize)
        .SetDispatchGroupSize(static_cast<uint32_t>(batch))
        .AddUniformVariables({{static_cast<uint32_t>(batch)},
                              {static_cast<uint32_t>(signal_length)},
                              {static_cast<uint32_t>(inner)},
                              {static_cast<uint32_t>(output_length)}});
    return context.RunProgram(program);
  }

  DFTDirectProgram program{transform_length, static_cast<uint32_t>(input_components),
                           static_cast<uint32_t>(output_components), is_inverse_, is_onesided_};
  program
      .AddInputs({{input_tensor, ProgramTensorMetadataDependency::Type}})
      .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::Type}})
      .CacheHint(std::to_string(transform_length), std::to_string(input_components),
                 std::to_string(output_components), std::to_string(is_inverse_), std::to_string(is_onesided_))
      .SetWorkgroupSize(kDftWorkgroupSize)
      .SetDispatchGroupSize(static_cast<uint32_t>(batch))
      .AddUniformVariables({{static_cast<uint32_t>(batch)},
                            {static_cast<uint32_t>(signal_length)},
                            {static_cast<uint32_t>(inner)},
                            {static_cast<uint32_t>(output_length)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
