// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

/**
 * Generates WebGPU shader code for reading zero points in quantized matrix multiplication
 *
 * @param nbits Number of bits for quantization (4 or 8)
 * @param has_zero_points Whether zero points are provided as an input
 * @param output_type Type name to use for zero point values in the generated code (default: "output_element_t")
 * @return String containing the generated WebGPU shader code
 */
std::string GenerateZeroPointReadingCode(uint32_t nbits, bool has_zero_points,
                                         const std::string& output_type = "output_element_t");

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
