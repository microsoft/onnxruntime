// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/matmul_compute_dispatcher.h"
#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

// ConvTranspose as GEMM + col2im (channels-last, group 1, 1D/2D).
//
// With the input viewed as X[M, C_in] (M = N * H * W) and the weights as W[C_in, kH * kW * C_out],
// cols = X * W holds, for every input pixel, its contribution to each of the kH x kW output pixels it
// scatters to. col2im then gathers, for every output pixel, the contributions of the input pixels that
// reach it, and adds the bias. The multiply-accumulates run on the tiled MatMul kernels.

// Whether a filter of this shape and element type, with these strides (empty means 1), takes the GEMM
// path: it needs enough taps per output, and the packed weights, in the compute type, must fit one buffer
// and one binding. Decided from the filter and the attributes alone so prepacking and compute agree.
bool UseConvTransposeGemm(const TensorShape& onnx_filter_shape, gsl::span<const int64_t> strides,
                          MLDataType filter_type, int64_t group, const wgpu::Limits& limits);

// Shape of the packed GEMM weights: [round_up(C_in, 4), round_up(kH * kW * C_out, 4)], zero padded.
TensorShape ConvTransposeGemmWeightShape(const TensorShape& onnx_filter_shape);

// Element type of the packed weights, the column buffer and the accumulation: f32 for float16 inputs.
MLDataType ConvTransposeGemmComputeType(MLDataType input_type);

// Packs ONNX weights [C_in, C_out, (kH,) kW] into the GEMM layout.
Status PackConvTransposeGemmWeight(ComputeContextBase& context, const Tensor& onnx_filter, Tensor& packed);

// Geometry of the (1D-lifted) 2D transposed convolution.
struct ConvTransposeGemmParams {
  uint32_t batch, in_height, in_width, in_channels;
  uint32_t out_height, out_width, out_channels;
  uint32_t kernel_height, kernel_width;
  uint32_t stride_h, stride_w, dilation_h, dilation_w;
  uint32_t pad_top, pad_left;
};

// `packed_filter` is the GEMM-layout weight tensor, or null to pack `onnx_filter` for this run. When the
// columns need several chunks and the f32 accumulator for a narrower output would not fit the device
// limits, the output is instead gathered in one pass from the input and the packed weights.
Status ComputeConvTransposeGemm(ComputeContext& context, MatMulComputeDispatcher& matmul,
                                const ConvTransposeGemmParams& params, const Tensor& input,
                                const Tensor* packed_filter, const Tensor& onnx_filter, const Tensor* bias,
                                Tensor& output);

class ConvTransposeGemmPackWeightProgram final : public Program<ConvTransposeGemmPackWeightProgram> {
 public:
  ConvTransposeGemmPackWeightProgram() : Program{"ConvTransposeGemmPackWeight"} {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"size", ProgramUniformVariableDataType::Uint32},
      {"in_channels", ProgramUniformVariableDataType::Uint32},
      {"out_channels", ProgramUniformVariableDataType::Uint32},
      {"kernel_height", ProgramUniformVariableDataType::Uint32},
      {"kernel_width", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"padded_cols", ProgramUniformVariableDataType::Uint32});
};

// Copies rows [row_offset, row_offset + rows) of X[M, K] into a [rows, padded_k] matrix, zero padded.
class ConvTransposeGemmPackInputProgram final : public Program<ConvTransposeGemmPackInputProgram> {
 public:
  ConvTransposeGemmPackInputProgram() : Program{"ConvTransposeGemmPackInput"} {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"size", ProgramUniformVariableDataType::Uint32},
      {"k", ProgramUniformVariableDataType::Uint32},
      {"padded_k", ProgramUniformVariableDataType::Uint32},
      {"row_offset", ProgramUniformVariableDataType::Uint32});
};

// Where col2im finds the sum of the earlier row chunks.
enum class Col2ImAccumulate {
  kNone,       // first chunk: start from the bias (or zero)
  kInPlace,    // add to the output written by the previous chunk
  kFromInput,  // add to a separate f32 "partial" input and write the final output
};

// Gathers the column rows [row_begin, row_end) into the output box of batches from n_begin, rows
// [oh_begin, oh_begin + region_height) and columns [ow_begin, ow_begin + region_width); `size` covers
// the box. An empty row range only initializes (or converts) the box. With `fused`, the inputs are X and
// the packed weights instead of the columns, and each column value is computed in place as the dot
// product of an input row and a weight column.
class ConvTransposeCol2ImProgram final : public Program<ConvTransposeCol2ImProgram> {
 public:
  ConvTransposeCol2ImProgram(bool has_bias, Col2ImAccumulate accumulate, bool fused = false)
      : Program{"ConvTransposeCol2Im"}, has_bias_{has_bias}, accumulate_{accumulate}, fused_{fused} {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"size", ProgramUniformVariableDataType::Uint32},
      {"out_channels", ProgramUniformVariableDataType::Uint32},
      {"out_width", ProgramUniformVariableDataType::Uint32},
      {"out_height", ProgramUniformVariableDataType::Uint32},
      {"in_width", ProgramUniformVariableDataType::Uint32},
      {"in_height", ProgramUniformVariableDataType::Uint32},
      {"kernel_width", ProgramUniformVariableDataType::Uint32},
      {"kernel_height", ProgramUniformVariableDataType::Uint32},
      {"stride_w", ProgramUniformVariableDataType::Uint32},
      {"stride_h", ProgramUniformVariableDataType::Uint32},
      {"dilation_w", ProgramUniformVariableDataType::Uint32},
      {"dilation_h", ProgramUniformVariableDataType::Uint32},
      {"pad_left", ProgramUniformVariableDataType::Uint32},
      {"pad_top", ProgramUniformVariableDataType::Uint32},
      {"row_begin", ProgramUniformVariableDataType::Uint32},
      {"row_end", ProgramUniformVariableDataType::Uint32},
      {"col_stride", ProgramUniformVariableDataType::Uint32},
      {"n_begin", ProgramUniformVariableDataType::Uint32},
      {"oh_begin", ProgramUniformVariableDataType::Uint32},
      {"ow_begin", ProgramUniformVariableDataType::Uint32},
      {"region_height", ProgramUniformVariableDataType::Uint32},
      {"region_width", ProgramUniformVariableDataType::Uint32},
      {"in_channels", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
  Col2ImAccumulate accumulate_;
  bool fused_;
};

}  // namespace webgpu
}  // namespace onnxruntime
