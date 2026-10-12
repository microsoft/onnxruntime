// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>

#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/configured_program.h"
#include "core/providers/webgpu/webgpu_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

#define WEBGPU_GATHER_BLOCK_QUANTIZED_PROGRAM_CONFIG(F) \
  F(bool, is_signed_)                                   \
  F(bool, is_uint8_)                                    \
  F(size_t, indices_rank_)                              \
  F(int, gather_axis_)                                  \
  F(int, bits_)                                         \
  F(bool, has_zeropoint_)                               \
  F(size_t, x_rank_)                                    \
  F(bool, is_fp_quantized_)                             \
  F(int32_t, fp_elem_type_)                             \
  F(std::string, scale_broadcast_axes_)

struct GatherBlockQuantizedProgramShader {
  WEBGPU_DECLARE_SPECIALIZATION(Config, WEBGPU_GATHER_BLOCK_QUANTIZED_PROGRAM_CONFIG);
  static Config MakeConfig(const bool is_signed, const bool is_uint8, size_t indices_rank, int gather_axis, int bits,
                           bool has_zeropoint, const TensorShape& x_shape, bool is_fp_quantized = false, int32_t fp_elem_type = 0,
                           std::string scale_broadcast_axes = {}) {
    Config result{};
    result.is_signed_ = bool{is_signed};
    result.is_uint8_ = bool{is_uint8};
    result.indices_rank_ = size_t{indices_rank};
    result.gather_axis_ = int{gather_axis};
    result.bits_ = int{bits};
    result.has_zeropoint_ = bool{has_zeropoint};
    result.x_rank_ = size_t{x_shape.NumDimensions()};
    result.is_fp_quantized_ = bool{is_fp_quantized};
    result.fp_elem_type_ = int32_t{fp_elem_type};
    result.scale_broadcast_axes_ = std::string{std::move(scale_broadcast_axes)};

    return result;
  }
  static constexpr std::string_view name = "GatherBlockQuantized";
  static Status GenerateShaderCode([[maybe_unused]] const Config& config, ConfiguredShaderHelper& sh);

#define WEBGPU_GATHER_BLOCK_QUANTIZED_PROGRAM_SHADER_UNIFORMS(U) \
  U(Uint32, output_size)                                         \
  U(Uint32, quantize_axis)                                       \
  U(Uint32, gather_axis)                                         \
  U(Uint32, block_size)                                          \
  U(Uint32, scale_qaxis_dim)                                     \
  U(Uint32, zp_packed_qaxis_dim)
  WEBGPU_DECLARE_UNIFORMS(WEBGPU_GATHER_BLOCK_QUANTIZED_PROGRAM_SHADER_UNIFORMS);
#undef WEBGPU_GATHER_BLOCK_QUANTIZED_PROGRAM_SHADER_UNIFORMS

  // When true, `data` holds FP8 or FP4 codes (rather than integer block-quantized codes) and
  // GenerateShaderCode emits a dequantization lookup table (indexed on the raw bit pattern)
  // instead of the (code - zero_point) integer formula. `fp_elem_type_` is the
  // ONNX_TENSOR_ELEMENT_DATA_TYPE_* value identifying which FP8/FP4 variant to build the table for.

  // Entry `i` is '1' when axis `i` of `scales` is broadcast (dim == 1 while `data`'s dim is > 1);
  // only possible for FP8/FP4 data on axes other than quantize_axis.
};
#undef WEBGPU_GATHER_BLOCK_QUANTIZED_PROGRAM_CONFIG

using GatherBlockQuantizedProgram = ConfiguredProgram<GatherBlockQuantizedProgramShader>;

class GatherBlockQuantized final : public WebGpuKernel {
 public:
  GatherBlockQuantized(const OpKernelInfo& info) : WebGpuKernel(info) {
    gather_axis_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("gather_axis", 0));
    block_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("block_size", 128));
    quantize_axis_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("quantize_axis", 1));
    bits_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("bits", 4));

    // block_size == 0 is only valid for FP8/FP4 `data`, which is validated (against the actual
    // input element type) in ComputeInternal, since the element type isn't known here.
    ORT_ENFORCE(block_size_ == 0 || (block_size_ >= 16 && ((block_size_ - 1) & block_size_) == 0),
                "'block_size' must be 0, or 2's power and not less than 16.");
  }
  Status ComputeInternal(ComputeContext& context) const override;

 private:
  int gather_axis_;
  int quantize_axis_;
  int block_size_;
  int bits_;
};

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
