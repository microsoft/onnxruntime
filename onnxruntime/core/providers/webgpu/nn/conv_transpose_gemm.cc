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

// Output box [n_begin, +batches) x [oh_begin, +height) x [ow_begin, +width), all output channels.
struct Col2ImRegion {
  uint32_t n_begin, batches, oh_begin, height, ow_begin, width;
};

// Output pixels [begin, begin + count) reached by input pixels [i0, i1] along one axis, where input i
// reaches outputs i * stride - pad through i * stride + (kernel - 1) * dilation - pad.
void ReachableSpan(int64_t i0, int64_t i1, uint32_t stride, uint32_t kernel, uint32_t dilation, uint32_t pad,
                   uint32_t out_size, uint32_t& begin, uint32_t& count) {
  const int64_t lo = std::max<int64_t>(0, i0 * stride - pad);
  const int64_t hi = std::min<int64_t>(int64_t{out_size} - 1,
                                       i1 * stride + int64_t{kernel - 1} * dilation - int64_t{pad});
  begin = static_cast<uint32_t>(std::min<int64_t>(lo, out_size));
  count = hi >= lo ? static_cast<uint32_t>(hi - lo + 1) : 0;
}

// Output box that the input rows [row_begin, row_end) of X[N * H * W, C_in] contribute to. Rows that
// span several input rows (or batches) cover whole output rows (or images).
Col2ImRegion ReachableRegion(const ConvTransposeGemmParams& p, uint64_t row_begin, uint64_t row_end) {
  Col2ImRegion region{0, p.batch, 0, p.out_height, 0, p.out_width};
  if (row_end <= row_begin) {
    region.batches = 0;
    return region;
  }
  const uint64_t image = uint64_t{p.in_height} * p.in_width;
  const uint64_t last = row_end - 1;
  region.n_begin = static_cast<uint32_t>(row_begin / image);
  region.batches = static_cast<uint32_t>(last / image) - region.n_begin + 1;
  if (region.batches > 1) {
    return region;
  }
  const int64_t ih0 = static_cast<int64_t>((row_begin % image) / p.in_width);
  const int64_t ih1 = static_cast<int64_t>((last % image) / p.in_width);
  ReachableSpan(ih0, ih1, p.stride_h, p.kernel_height, p.dilation_h, p.pad_top, p.out_height, region.oh_begin,
                region.height);
  if (ih0 == ih1) {
    ReachableSpan(static_cast<int64_t>(row_begin % p.in_width), static_cast<int64_t>(last % p.in_width), p.stride_w,
                  p.kernel_width, p.dilation_w, p.pad_left, p.out_width, region.ow_begin, region.width);
  }
  return region;
}

}  // namespace

MLDataType ConvTransposeGemmComputeType(MLDataType input_type) {
  return input_type == DataTypeImpl::GetType<MLFloat16>() ? DataTypeImpl::GetType<float>() : input_type;
}

bool UseConvTransposeGemm(const TensorShape& onnx_filter_shape, MLDataType filter_type, int64_t group,
                          const wgpu::Limits& limits) {
  const size_t rank = onnx_filter_shape.NumDimensions();
  if (group != 1 || (rank != 3 && rank != 4) || onnx_filter_shape[0] < kGemmMinInputChannels ||
      onnx_filter_shape.Size() <= 0) {
    return false;
  }
  // The packed weights are padded and, for float16, promoted to f32; the direct path keeps the original size.
  const auto round_up4 = [](uint64_t v) { return (v + 3) / 4 * 4; };
  const uint64_t packed_bytes = round_up4(static_cast<uint64_t>(onnx_filter_shape[0])) *
                                round_up4(static_cast<uint64_t>(onnx_filter_shape.SizeFromDimension(1))) *
                                ConvTransposeGemmComputeType(filter_type)->Size();
  return packed_bytes <= std::min<uint64_t>(limits.maxBufferSize, limits.maxStorageBufferBindingSize);
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
                            << "  v = packed_value_t(" << w.GetByOffset("((r * uniforms.out_channels + co) * uniforms.kernel_height + kh) * uniforms.kernel_width + kw") << ");\n"
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
                            << "  v = a_value_t(" << x.GetByOffset("(uniforms.row_offset + r) * uniforms.k + c") << ");\n"
                            << "}\n"
                            << a.SetByOffset("global_idx", "v") << "\n";
  return Status::OK();
}

Status ConvTransposeCol2ImProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Fused: "cols" is X[M, in_channels] and "weights" the packed [*, col_stride] weights.
  const auto& cols = shader.AddInput("cols", ShaderUsage::UseUniform);
  const ShaderVariableHelper* weights = fused_ ? &shader.AddInput("weights", ShaderUsage::UseUniform) : nullptr;
  const ShaderVariableHelper* bias = has_bias_ ? &shader.AddInput("bias", ShaderUsage::UseUniform) : nullptr;
  const ShaderVariableHelper* partial =
      accumulate_ == Col2ImAccumulate::kFromInput ? &shader.AddInput("partial", ShaderUsage::UseUniform) : nullptr;
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
       << "let ow = uniforms.ow_begin + pixel % uniforms.region_width;\n"
       << "let nh = pixel / uniforms.region_width;\n"
       << "let oh = uniforms.oh_begin + nh % uniforms.region_height;\n"
       << "let n = uniforms.n_begin + nh / uniforms.region_height;\n"
       << "let out_idx = ((n * uniforms.out_height + oh) * uniforms.out_width + ow) * uniforms.out_channels + co;\n";
  if (partial != nullptr) {
    body << "var acc = f32(" << partial->GetByOffset("out_idx") << ");\n";
  } else if (accumulate_ == Col2ImAccumulate::kInPlace) {
    body << "var acc = f32(" << output.GetByOffset("out_idx") << ");\n";
  } else if (bias != nullptr) {
    body << "var acc = f32(" << bias->GetByOffset("co") << ");\n";
  } else {
    body << "var acc = 0.0;\n";
  }
  body << "if (uniforms.row_begin < uniforms.row_end) {\n"
       << taps("h", "oh", "stride_h", "dilation_h", "pad_top", "kernel_height", "in_height", "")
       << taps("w", "ow", "stride_w", "dilation_w", "pad_left", "kernel_width", "in_width", "  ")
       << "    let m = (n * uniforms.in_height + ih) * uniforms.in_width + iw;\n"
       << "    if (m >= uniforms.row_begin && m < uniforms.row_end) {\n"
       << "      let tap_col = (kh * uniforms.kernel_width + kw) * uniforms.out_channels + co;\n";
  if (fused_) {
    body << "      for (var ci = 0u; ci < uniforms.in_channels; ci++) {\n"
         << "        acc += f32(" << cols.GetByOffset("m * uniforms.in_channels + ci") << ") * f32("
         << weights->GetByOffset("ci * uniforms.col_stride + tap_col") << ");\n"
         << "      }\n";
  } else {
    body << "      acc += f32(" << cols.GetByOffset("(m - uniforms.row_begin) * uniforms.col_stride + tap_col")
         << ");\n";
  }
  body << "    }\n"
       << "  }\n"
       << "}\n"
       << "}\n"
       << output.SetByOffset("out_idx", "output_value_t(acc)") << "\n";
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

  const bool weights_are_constant = packed_filter != nullptr;
  // float16 runs the GEMM and the accumulation in f32: a per-tap partial sum can overflow or lose the
  // precision that later taps would cancel.
  const MLDataType compute_type = ConvTransposeGemmComputeType(input.DataType());
  Tensor packed_for_run;
  if (packed_filter == nullptr) {
    packed_for_run = context.CreateGPUTensor(compute_type, ConvTransposeGemmWeightShape(onnx_filter.Shape()));
    ORT_RETURN_IF_ERROR(PackConvTransposeGemmWeight(context, onnx_filter, packed_for_run));
    packed_filter = &packed_for_run;
  }

  // Rows of X per GEMM, bounded so the column buffer and the padded input chunk fit in one binding.
  const uint64_t element_size = compute_type->Size();
  const uint64_t max_bytes = std::min<uint64_t>(kMaxChunkBytes, context.DeviceLimits().maxStorageBufferBindingSize);
  const uint64_t row_bytes = element_size * std::max(padded_cols, padded_k);
  const uint32_t chunk_rows = narrow<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(m, max_bytes / row_bytes)));
  const bool pack_input = padded_k != k || chunk_rows < m || compute_type != input.DataType();
  // With several chunks, a first pass writes the bias (or zeros) to the whole output, each chunk adds its
  // rows into the output box it can reach, and a narrower output type accumulates in an f32 buffer that a
  // last pass converts.
  const bool multi_chunk = chunk_rows < m;
  const bool use_partial = multi_chunk && compute_type != output.DataType();
  const auto& limits = context.DeviceLimits();
  if (use_partial && static_cast<uint64_t>(output.Shape().Size()) * element_size >
                         std::min<uint64_t>(limits.maxBufferSize, limits.maxStorageBufferBindingSize)) {
    // The f32 accumulator would not fit: gather every output from X and the weights in one pass instead.
    ConvTransposeCol2ImProgram fused{bias != nullptr, Col2ImAccumulate::kNone, /*fused=*/true};
    const uint32_t size = narrow<uint32_t>(output.Shape().Size());
    fused.AddInput({&input, ProgramTensorMetadataDependency::Type})
        .AddInput({packed_filter, ProgramTensorMetadataDependency::Type});
    if (bias != nullptr) {
      fused.AddInput({bias, ProgramTensorMetadataDependency::Type});
    }
    fused.AddOutput({&output, ProgramTensorMetadataDependency::Type})
        .CacheHint(bias != nullptr, static_cast<int>(Col2ImAccumulate::kNone), "fused")
        .SetDispatchGroupSize(Dispatch(size))
        .AddUniformVariables({{size},
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
                              {0u},
                              {narrow<uint32_t>(m)},
                              {padded_cols},
                              {0u},
                              {0u},
                              {0u},
                              {p.out_height},
                              {p.out_width},
                              {k}});
    return context.RunProgram(fused);
  }

  Tensor a_buffer;
  if (pack_input) {
    a_buffer = context.CreateGPUTensor(compute_type, TensorShape({chunk_rows, padded_k}));
  }
  Tensor col_buffer = context.CreateGPUTensor(compute_type, TensorShape({chunk_rows, padded_cols}));
  Tensor partial_sums;
  if (use_partial) {
    partial_sums = context.CreateGPUTensor(compute_type, output.Shape());
  }
  Tensor& sums = use_partial ? partial_sums : output;
  const Activation no_activation{};
  const Col2ImRegion full_region{0, p.batch, 0, p.out_height, 0, p.out_width};

  auto run_col2im = [&](const Tensor& col, uint64_t row_begin, uint64_t row_end, const Col2ImRegion& region,
                        bool with_bias, Col2ImAccumulate accumulate, Tensor& target) -> Status {
    const uint32_t size = narrow<uint32_t>(uint64_t{region.batches} * region.height * region.width * p.out_channels);
    if (size == 0) {
      return Status::OK();
    }
    ConvTransposeCol2ImProgram col2im{with_bias, accumulate};
    col2im.AddInput({&col, ProgramTensorMetadataDependency::Type});
    if (with_bias) {
      col2im.AddInput({bias, ProgramTensorMetadataDependency::Type});
    }
    if (accumulate == Col2ImAccumulate::kFromInput) {
      col2im.AddInput({&partial_sums, ProgramTensorMetadataDependency::Type});
    }
    col2im.AddOutput({&target, ProgramTensorMetadataDependency::Type})
        .CacheHint(with_bias, static_cast<int>(accumulate))
        .SetDispatchGroupSize(Dispatch(size))
        .AddUniformVariables({{size},
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
                              {narrow<uint32_t>(row_end)},
                              {padded_cols},
                              {region.n_begin},
                              {region.oh_begin},
                              {region.ow_begin},
                              {region.height},
                              {region.width},
                              {k}});
    return context.RunProgram(col2im);
  };

  if (multi_chunk) {
    ORT_RETURN_IF_ERROR(run_col2im(col_buffer, 0, 0, full_region, bias != nullptr, Col2ImAccumulate::kNone, sums));
  }
  // An empty spatial input (m == 0) still runs one col2im, which reads no columns and writes the bias
  // (or zeros) to the non-empty output.
  uint64_t row_begin = 0;
  do {
    const uint32_t rows = narrow<uint32_t>(std::min<uint64_t>(chunk_rows, m - row_begin));
    Tensor col = Prefix(col_buffer, TensorShape({std::max(rows, 1u), padded_cols}));
    if (rows > 0) {
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
      ORT_RETURN_IF_ERROR(matmul.Compute(context, no_activation, {&a, packed_filter}, &col,
                                         /*is_channels_last=*/true, weights_are_constant));
    }
    if (multi_chunk) {
      ORT_RETURN_IF_ERROR(run_col2im(col, row_begin, row_begin + rows, ReachableRegion(p, row_begin, row_begin + rows),
                                     false, Col2ImAccumulate::kInPlace, sums));
    } else {
      ORT_RETURN_IF_ERROR(run_col2im(col, 0, rows, full_region, bias != nullptr, Col2ImAccumulate::kNone, output));
    }
    row_begin += rows;
  } while (row_begin < m);
  if (use_partial) {
    ORT_RETURN_IF_ERROR(run_col2im(col_buffer, 0, 0, full_region, false, Col2ImAccumulate::kFromInput, output));
  }
  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
