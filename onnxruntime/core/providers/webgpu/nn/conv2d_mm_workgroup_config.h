// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace webgpu {

/**
 * Chooses the Conv2dMM workgroup y dimension and the matching elements-per-thread.
 *
 * Scoped to the Conv2d kernel rather than shared: MatMul has its own copy of this rule, so the
 * two can diverge without either becoming a general-purpose utility.
 *
 * Takes the adapter and limit values as plain scalars rather than wgpu::AdapterInfo and
 * wgpu::Limits so that the eligibility decision can be unit tested -- wgpu::AdapterInfo has
 * const members and cannot be constructed by a test. Keeping wgpu out of this header also keeps
 * it free of the generated WGSL headers that shader_helper.h pulls in.
 *
 * On return, `workgroup_size_y * elements_per_thread_y` covers the same rows of A as the
 * default configuration, so the caller's dispatch grid is unaffected.
 */
void SelectConv2dMMWorkgroupConfig(uint32_t subgroup_min_size,
                                   uint32_t max_compute_workgroup_size_y,
                                   uint32_t max_compute_invocations_per_workgroup,
                                   bool is_nvidia,
                                   uint32_t target_workgroup_size,
                                   bool is_vec4,
                                   int64_t in_channels,
                                   uint32_t dim_a_outer,
                                   uint32_t& workgroup_size_y,
                                   int64_t& elements_per_thread_y);

}  // namespace webgpu
}  // namespace onnxruntime
