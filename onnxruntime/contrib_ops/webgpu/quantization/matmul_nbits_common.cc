// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"

#include <sstream>

#include "core/common/common.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

std::string GenerateZeroPointReadingCode(uint32_t nbits, bool has_zero_points,
                                         const std::string& output_type) {
  ORT_ENFORCE(nbits == 8 || nbits == 4, "Only 4/8 bits are supported for webgpu matmulnbits");
  std::stringstream ss;

  if (has_zero_points) {
    ss << "const elements_in_uint32 = " << (32 / nbits) << "u;\n"
       << "const bits = " << nbits << "u;\n";
    ss << R"(
fn mm_read_zero(row : u32, col : u32, r_dim: u32, c_dim: u32) -> )"
       << output_type << R"( {
  if (row < r_dim && col < c_dim) {
    let offset = row * c_dim + col;

    // u32 holds elements_in_uint32 packed nbits.
    let array_index = offset / elements_in_uint32;
    let component_index = offset % elements_in_uint32;
    let packed_value = zero_points[array_index];

    // Extract the nbits component
    let shift_amount = component_index * bits;
)";
    ss << "    let masked_value = (packed_value >> shift_amount) & " << (nbits == 4 ? "0xFu" : "0xFF") << ";\n";
    ss << R"(
    return )"
       << output_type << R"((masked_value);
  }
  return )"
       << output_type << R"((0);
}
)";
  } else {
    ss << "const default_zero_point = " << (nbits == 4 ? 8 : 128) << ";\n";
    ss << R"(
fn mm_read_zero(row : u32, col : u32, r_dim: u32, c_dim: u32) -> )"
       << output_type << R"( {
  return )"
       << output_type << R"((default_zero_point);
}
)";
  }

  return ss.str();
}

bool HasDP4ADeviceSupport(int context_id) {
  auto& ctx = onnxruntime::webgpu::WebGpuContextFactory::GetContext(context_id);
  return ctx.DeviceHasFeature(wgpu::FeatureName::Subgroups) &&
         ctx.AdapterInfo().vendor != std::string_view{"apple"};
}

bool CanApplyWideTileMatMulNBits(uint32_t M,
                                 uint32_t K,
                                 uint32_t block_size,
                                 int64_t nbits,
                                 bool has_weight_idx_indirect,
                                 uint32_t components_a,
                                 uint32_t components_b) {
  if (has_weight_idx_indirect) {
    return false;
  }

  // If not provided, calculate components_a and components_b.
  if (components_a == 0) {
    components_a = onnxruntime::webgpu::GetMaxComponents(K);
  }
  if (components_b == 0) {
    const uint32_t block_size_per_col = block_size;
    const uint32_t blob_size = (block_size_per_col / 8) * static_cast<uint32_t>(nbits);
    const uint32_t blob_size_in_words = blob_size / 4;
    components_b = onnxruntime::webgpu::GetMaxComponents(blob_size_in_words);
  }

  return block_size == 32 &&
         components_a == 4 &&
         components_b == 4 &&
         nbits != 2 &&
         M >= kMinMForTileOptimization;
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
