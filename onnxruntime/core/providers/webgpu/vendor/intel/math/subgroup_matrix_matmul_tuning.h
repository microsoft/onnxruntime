// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(__wasm__)

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Lightweight tuning interface for the Intel subgroup-matrix MatMul kernel.
// Kept free of the WebGPU shader/program headers so offline-tuning tools and
// tests can use it without pulling in the generated WGSL template machinery.

namespace onnxruntime {
namespace webgpu {
namespace intel {

// Tile + split-K configuration for one MatMul problem.
struct SgMatMulConfig {
  uint32_t tile_m;   // output rows per workgroup (multiple of kSubgroupMatrixM)
  uint32_t tile_n;   // output cols per workgroup (multiple of kSubgroupMatrixN)
  uint32_t split_k;  // subgroups cooperating along K (1 = no split)
};

// Offline-autotuning hooks. Not used on the production inference path.
//
// EnumerateSgMatMulConfigs returns every candidate config that is valid for the
// given problem (passes the same constraints the runtime selector enforces), so
// a tuner can benchmark each one.
std::vector<SgMatMulConfig> EnumerateSgMatMulConfigs(uint32_t M, uint32_t N, uint32_t K);

// SetSgMatMulConfigOverride pins a config that takes precedence over both the
// ORT_WEBGPU_SGMM_FORCE env var and the built-in heuristic (still re-validated
// per problem). Pass std::nullopt to clear. Intended for the autotuner only and
// is not thread-safe; set it while no MatMul is running.
void SetSgMatMulConfigOverride(std::optional<SgMatMulConfig> config);

// SetSgMatMulDisabled forces the subgroup-matrix kernel to decline every MatMul
// so the generic WebGPU MatMul path runs instead. The autotuner uses this to
// measure the default implementation as a baseline. Not thread-safe; set it
// while no MatMul is running.
void SetSgMatMulDisabled(bool disabled);

// GetSgMatMulDeviceArch returns the GPU architecture string (e.g. "xe-2lpg") of
// the device that ran the most recent MatMul, or an empty string if none has
// run yet. The autotuner uses it to label its CSV so the codegen script can
// build a separate tuned table per architecture.
std::string GetSgMatMulDeviceArch();

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
