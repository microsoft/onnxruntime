// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <cstdint>
#include <string_view>

#include "core/providers/webgpu/compute_context.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

// Approximate number of subgroups an Intel Xe GPU keeps resident at once. Each Xe
// core runs 4 XVE x 8 SIMD32 hardware threads = 32 subgroups, so the count is
// (number of Xe cores) x 32. Returns 0 for an unrecognized architecture so each
// caller can apply its own fallback policy.
uint32_t HwSubgroups(std::string_view arch);

// Returns true if the device exposes an m x n x k F16/F16 subgroup matrix config
// within the Intel Xe subgroup size range (min 16, max 32).
bool IsSubgroupMatrixConfigSupported(const ComputeContextBase& context, uint32_t m, uint32_t n, uint32_t k);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
