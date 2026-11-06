// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace onnxruntime {
namespace webgpu {
namespace intel {

namespace gpu_arch {
inline constexpr std::string_view kGen12Hp = "gen-12hp";
inline constexpr std::string_view kXeLpg = "xe-lpg";
inline constexpr std::string_view kXe3Lpg = "xe-3lpg";
}  // namespace gpu_arch

bool ShouldFoldMatMulSubgroupBatch(std::string_view architecture,
                                   int64_t m,
                                   bool can_fold_batch_into_m);

int64_t MatMulSubgroupElementsPerThreadY(std::string_view architecture, uint32_t m);

bool CanUseAVec4CooperativeLoad(std::string_view architecture,
                                uint32_t dim_inner,
                                int64_t elements_per_thread_y);

std::optional<uint32_t> SelectMatMulSubgroupSize(uint32_t adapter_min_subgroup_size,
                                                 uint32_t adapter_max_subgroup_size,
                                                 bool has_subgroup_size_control);

// Same as above, but prefers kMatMulSubgroup16Size on architectures that use the subgroup-16 kernel.
std::optional<uint32_t> SelectMatMulSubgroupSize(std::string_view architecture,
                                                 uint32_t adapter_min_subgroup_size,
                                                 uint32_t adapter_max_subgroup_size,
                                                 bool has_subgroup_size_control);

inline constexpr uint32_t kMatMulSubgroup16Size = 16;

// Whether the subgroup MatMul/Gemm uses the subgroup-16 kernel (gemm_subgroup16.h) instead of the default one.
bool UseMatMulSubgroup16Kernel(std::string_view architecture, uint32_t subgroup_size);

// Rows of the output each invocation of the subgroup-16 kernel computes for an [m, k] x [k, n] product repeated
// `batch_size` times. `f32_accumulator` is true unless the product accumulates in f16.
int64_t MatMulSubgroup16ElementsPerThreadY(uint32_t m, uint32_t n, uint64_t batch_size,
                                           bool f32_accumulator, bool a_vec4);

// The subgroup-16 kernel loads A as vec4 values along K when K is a multiple of 4.
inline bool CanLoadMatMulSubgroup16AVec4(uint64_t dim_inner) {
  return dim_inner % 4 == 0;
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
