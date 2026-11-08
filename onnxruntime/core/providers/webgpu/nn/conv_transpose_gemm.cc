// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/conv_transpose_gemm.h"

#include <algorithm>
#include <vector>

#include "core/providers/webgpu/nn/fuse_utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Below this many input channels the direct kernel's per-output loop is short and the column buffer
// round trip does not pay off.
constexpr int64_t kGemmMinInputChannels = 16;
// Upper bound on one column (and padded-input) chunk, kept well below common binding size limits.
constexpr uint64_t kMaxChunkBytes = uint64_t{256} << 20;

uint32_t RoundUp4(uint32_t v) { return (v + 3) / 4 * 4; }

// A smaller view at the start of `tensor`'s buffer (CreateTensorView requires equal sizes).
Tensor Prefix(const Tensor& tensor, const TensorShape& shape) {
  return {tensor.DataType(), shape, const_cast<void*>(tensor.DataRaw()), tensor.Location()};
}

uint32_t Dispatch(uint32_t size) { return (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE; }

}  // namespace

bool UseConvTransposeGemm(const TensorShape& onnx_filter_shape, int64_t group) {
  const size_t rank = onnx_filter_shape.NumDimensions();
  return group == 1 && (rank == 3 || rank == 4) && onnx_filter_shape[0] >= kGemmMinInputChannels &&
         onnx_filter_shape.Size() > 0;
}

TensorShape ConvTransposeGemmWeightShape(const TensorShape& onnx_filter_shape) {
  const int64_t cols = onnx_filter_shape.SizeFromDimension(1);  // C_out * kH * kW
  return TensorShape({RoundUp4(narrow<uint32_t>(onnx_filter_shape[0])), RoundUp4(narrow<uint32_t>(cols))});
}

Status ConvTransposeGemmPackWeightProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& w = shader.AddInput("w", ShaderUsage::UseUniform);
  const auto& packed = shader.AddOutput("packed", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.size")
                            << "let r = global_idx / uniforms.padded_cols;\n"
                            << "let c = global_idx % uniforms.padded_cols;\n"
                            << "var v = packed_value_t(0);\n"
                            << "if (r < uniforms.in_channels && c < uniforms.cols) {\n"
                            << "  let kk = c / uniforms.out_channels;\n"
                            << "  let co = c % uniforms.out_channels;\n"
                            << "  let kh = kk / uniforms.kernel_width;\n"
                            << "  let kw = kk % uniforms.kernel_width;\n"
                            << "  v = " << w.GetByOffset("((r * uniforms.out_channels + co) * uniforms.kernel_height + kh) * uniforms.kernel_width + kw") << ";\n"
                            << "}\n"
                            << packed.SetByOffset("global_idx", "v") << "\n";
  return Status::OK();
}

Status ConvTransposeGemmPackInputProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& x = shader.AddInput("x", ShaderUsage::UseUniform);
  const auto& a = shader.AddOutput("a", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.size")
                            << "let r = global_idx / uniforms.padded_k;\n"
                            << "let c = global_idx % uniforms.padded_k;\n"
                            << "var v = a_value_t(0);\n"
                            << "if (c < uniforms.k) {\n"
                            << "  v = " << x.GetByOffset("(uniforms.row_offset + r) * uniforms.k + c") << ";\n"
                            << "}\n"
                            << a.SetByOffset("global_idx", "v") << "\n";
  return Status::OK();
}

Status ConvTransposeCol2ImProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& cols = shader.AddInput("cols", ShaderUsage::UseUniform);
  const ShaderVariableHelper* bias = has_bias_ ? &shader.AddInput("bias", ShaderUsage::UseUniform) : nullptr;
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  // Output pixel o receives tap k of input pixel i when o = i * stride + k * dilation - pad. Taps are
  // visited in increasing k, so the numerator o + pad - k * dilation only decreases: once negative, stop.
  // With unit dilation only every stride-th tap can land on o, so the loop steps over the others.
  auto taps = [](const std::string& axis, const std::string& o, const std::string& stride,
                 const std::string& dilation, const std::string& pad, const std::string& kernel,
                 const std::string& in_size, const std::string& indent) {
    std::string s;
    s += indent + "let num_" + axis + "0 = " + o + " + uniforms." + pad + ";\n";
    s += indent + "let step_" + axis + " = select(1u, uniforms." + stride + ", uniforms." + dilation + " == 1u);\n";
    s += indent + "var k" + axis + " = select(0u, num_" + axis + "0 % uniforms." + stride + ", uniforms." + dilation +
         " == 1u);\n";
    s += indent + "for (; k" + axis + " < uniforms." + kernel + "; k" + axis + " += step_" + axis + ") {\n";
    s += indent + "  let tap_" + axis + " = k" + axis + " * uniforms." + dilation + ";\n";
    s += indent + "  if (tap_" + axis + " > num_" + axis + "0) { break; }\n";
    s += indent + "  let num_" + axis + " = num_" + axis + "0 - tap_" + axis + ";\n";
    s += indent + "  if (num_" + axis + " % uniforms." + stride + " != 0u) { continue; }\n";
    s += indent + "  let i" + axis + " = num_" + axis + " / uniforms." + stride + ";\n";
    s += indent + "  if (i" + axis + " >= uniforms." + in_size + ") { continue; }\n";
    return s;
  };

  auto& body = shader.MainFunctionBody();
  body << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.size")
       << "let co = global_idx % uniforms.out_channels;\n"
       << "let pixel = global_idx / uniforms.out_channels;\n"
       << "let ow = pixel % uniforms.out_width;\n"
       << "let nh = pixel / uniforms.out_width;\n"
       << "let oh = nh % uniforms.out_height;\n"
       << "let n = nh / uniforms.out_height;\n";
  if (accumulate_) {
    body << "var acc = f32(" << output.GetByOffset("global_idx") << ");\n";
  } else if (bias != nullptr) {
    body << "var acc = f32(" << bias->GetByOffset("co") << ");\n";
  } else {
    body << "var acc = 0.0;\n";
  }
  body << taps("h", "oh", "stride_h", "dilation_h", "pad_top", "kernel_height", "in_height", "")
       << taps("w", "ow", "stride_w", "dilation_w", "pad_left", "kernel_width", "in_width", "  ")
       << "    let m = (n * uniforms.in_height + ih) * uniforms.in_width + iw;\n"
       << "    if (m >= uniforms.row_begin && m < uniforms.row_end) {\n"
       << "      let col = (m - uniforms.row_begin) * uniforms.col_stride + (kh * uniforms.kernel_width + kw) * "
          "uniforms.out_channels + co;\n"
       << "      acc += f32(" << cols.GetByOffset("col") << ");\n"
       << "    }\n"
       << "  }\n"
       << "}\n"
       << output.SetByOffset("global_idx", "output_value_t(acc)") << "\n";
  return Status::OK();
}

Status PackConvTransposeGemmWeight(ComputeContextBase& context, const Tensor& onnx_filter, Tensor& packed) {
  const TensorShape& shape = onnx_filter.Shape();
  const bool is_1d = shape.NumDimensions() == 3;
  const uint32_t kernel_height = is_1d ? 1u : narrow<uint32_t>(shape[2]);
  const uint32_t kernel_width = narrow<uint32_t>(shape[shape.NumDimensions() - 1]);
  const uint32_t size = narrow<uint32_t>(packed.Shape().Size());
  ConvTransposeGemmPackWeightProgram program;
  program.AddInput({&onnx_filter, ProgramTensorMetadataDependency::Type})
      .AddOutput({&packed, ProgramTensorMetadataDependency::Type})
      .SetDispatchGroupSize(Dispatch(size))
      .AddUniformVariables({{size},
                            {narrow<uint32_t>(shape[0])},
                            {narrow<uint32_t>(shape[1])},
                            {kernel_height},
                            {kernel_width},
                            {narrow<uint32_t>(shape.SizeFromDimension(1))},
                            {narrow<uint32_t>(packed.Shape()[1])}});
  return context.RunProgram(program);
}

Status ComputeConvTransposeGemm(ComputeContext& context, MatMulComputeDispatcher& matmul,
                                const ConvTransposeGemmParams& p, const Tensor& input,
                                const Tensor* packed_filter, const Tensor& onnx_filter, const Tensor* bias,
                                Tensor& output) {
  const uint32_t k = p.in_channels;
  const uint32_t padded_k = RoundUp4(k);
  const uint32_t cols = p.kernel_height * p.kernel_width * p.out_channels;
  const uint32_t padded_cols = RoundUp4(cols);
  const uint64_t m = uint64_t{p.batch} * p.in_height * p.in_width;
  const uint32_t output_size = narrow<uint32_t>(output.Shape().Size());

  const bool weights_are_constant = packed_filter != nullptr;
  Tensor packed_for_run;
  if (packed_filter == nullptr) {
    packed_for_run = context.CreateGPUTensor(input.DataType(), ConvTransposeGemmWeightShape(onnx_filter.Shape()));
    ORT_RETURN_IF_ERROR(PackConvTransposeGemmWeight(context, onnx_filter, packed_for_run));
    packed_filter = &packed_for_run;
  }

  // Rows of X per GEMM, bounded so the column buffer and the padded input chunk fit in one binding.
  const uint64_t element_size = input.DataType()->Size();
  const uint64_t max_bytes = std::min<uint64_t>(kMaxChunkBytes, context.DeviceLimits().maxStorageBufferBindingSize);
  const uint64_t row_bytes = element_size * std::max(padded_cols, padded_k);
  const uint32_t chunk_rows = narrow<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(m, max_bytes / row_bytes)));
  const bool pack_input = padded_k != k || chunk_rows < m;

  Tensor a_buffer;
  if (pack_input) {
    a_buffer = context.CreateGPUTensor(input.DataType(), TensorShape({chunk_rows, padded_k}));
  }
  Tensor col_buffer = context.CreateGPUTensor(input.DataType(), TensorShape({chunk_rows, padded_cols}));
  const Activation no_activation{};

  for (uint64_t row_begin = 0; row_begin < m; row_begin += chunk_rows) {
    const uint32_t rows = narrow<uint32_t>(std::min<uint64_t>(chunk_rows, m - row_begin));
    Tensor a;
    if (pack_input) {
      a = Prefix(a_buffer, TensorShape({rows, padded_k}));
      const uint32_t size = rows * padded_k;
      ConvTransposeGemmPackInputProgram pack;
      pack.AddInput({&input, ProgramTensorMetadataDependency::Type})
          .AddOutput({&a, ProgramTensorMetadataDependency::Type})
          .SetDispatchGroupSize(Dispatch(size))
          .AddUniformVariables({{size}, {k}, {padded_k}, {narrow<uint32_t>(row_begin)}});
      ORT_RETURN_IF_ERROR(context.RunProgram(pack));
    } else {
      a = CreateTensorView(input, TensorShape({rows, k}));
    }
    Tensor col = Prefix(col_buffer, TensorShape({rows, padded_cols}));
    ORT_RETURN_IF_ERROR(matmul.Compute(context, no_activation, {&a, packed_filter}, &col,
                                       /*is_channels_last=*/true, weights_are_constant));

    const bool accumulate = row_begin > 0;
    ConvTransposeCol2ImProgram col2im{bias != nullptr, accumulate};
    col2im.AddInput({&col, ProgramTensorMetadataDependency::Type});
    if (bias != nullptr) {
      col2im.AddInput({bias, ProgramTensorMetadataDependency::Type});
    }
    col2im.AddOutput({&output, ProgramTensorMetadataDependency::Type})
        .CacheHint(bias != nullptr, accumulate)
        .SetDispatchGroupSize(Dispatch(output_size))
        .AddUniformVariables({{output_size},
                              {p.out_channels},
                              {p.out_width},
                              {p.out_height},
                              {p.in_width},
                              {p.in_height},
                              {p.kernel_width},
                              {p.kernel_height},
                              {p.stride_w},
                              {p.stride_h},
                              {p.dilation_w},
                              {p.dilation_h},
                              {p.pad_left},
                              {p.pad_top},
                              {narrow<uint32_t>(row_begin)},
                              {narrow<uint32_t>(row_begin + rows)},
                              {padded_cols}});
    ORT_RETURN_IF_ERROR(context.RunProgram(col2im));
  }
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
