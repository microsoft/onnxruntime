// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include <sstream>
#include "core/common/common.h"

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

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
