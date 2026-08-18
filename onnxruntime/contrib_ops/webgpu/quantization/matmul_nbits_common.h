// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <cstdint>

namespace onnxruntime {
class Tensor;

namespace webgpu {
class ComputeContext;
}  // namespace webgpu
}  // namespace onnxruntime

namespace onnxruntime {
namespace contrib {
namespace webgpu {

inline constexpr uint32_t kMinMForTileOptimization = 4u;

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

/// Returns true when the default WebGPU device supports the DP4A kernel path
/// (Subgroups feature present and non-Apple vendor).
/// \p context_id is the WebGpuContext slot (0 for the default context).
bool HasDP4ADeviceSupport(int context_id = 0);

// Feasibility + dispatch-precondition check for the wide-tile MatMulNBits
// kernel (Block32 / fp16 a4-component prefill). Returns true when the kernel
// is supported for the given dims and the M-threshold is met. Optional components_a
// and components_b parameters allow skipping recalculation if already available.
bool CanApplyWideTileMatMulNBits(uint32_t M,
                                 uint32_t K,
                                 uint32_t block_size,
                                 int64_t nbits,
                                 bool has_weight_idx_indirect = false,
                                 uint32_t components_a = 0,
                                 uint32_t components_b = 0);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
