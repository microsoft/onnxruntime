// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)

#include <cstddef>
#include <cstdint>
#include <optional>

#include <cuda_runtime_api.h>
#include <gsl/span>

#include "core/common/inlined_containers.h"
#include "core/framework/level1_memory_estimate.h"
#include "core/framework/workspace_input_shape.h"
#include "core/framework/workspace_requirement.h"
#include "contrib_ops/cuda/bert/attention_kernel_options.h"
#include "contrib_ops/cuda/bert/group_query_attention_workspace_bounds.h"

namespace onnxruntime {
// Do not forward-declare Node. In-tree and shared-provider headers define
// different Node class keys; translation units must supply their own world.
namespace contrib {
namespace cuda {

struct GQAWorkspaceEstimateConfig {
  size_t qkv_element_size = 0;
  size_t cache_element_size = 0;
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  int64_t causal = 1;
  int64_t local_window_size = -1;
  bool sliding_window_cache = false;
  // Required for non-windowed estimation because WorkspaceInputShape does not
  // carry the total_sequence_length scalar value.
  int64_t max_total_sequence_length = 0;
  bool do_rotary = false;
  bool smooth_softmax = false;
  float softcap = 0.0f;
  GQAKvQuantizationType k_quantization = GQAKvQuantizationType::None;
  GQAKvQuantizationType v_quantization = GQAKvQuantizationType::None;
  int64_t kv_cache_bit_width = 0;
  bool is_bf16 = false;
  bool cache_is_fp8 = false;
  bool enable_xqa = true;
  bool disable_flash_decode = false;
  // Exact kernel state; true omits per-run conversion scratch.
  bool head_sink_is_prepacked = false;
  // Level-1 possibility; true charges persistent and initialization lifetimes.
  bool head_sink_may_be_prepacked = false;
};

std::optional<GQAWorkspaceAggregate> EstimateGroupQueryAttentionWorkspace(
    const GQAWorkspaceEstimateConfig& config,
    gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options);

std::optional<GQAWorkspaceAggregate> EstimateGroupQueryAttentionWorkspace(
    const Node& node,
    gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options,
    bool head_sink_is_constant_initializer = false);

void SetGroupQueryAttentionWorkspaceRequirements(
    const GQAWorkspaceAggregate& estimate,
    InlinedVector<WorkspaceRequirement>& requirements);

void SetGroupQueryAttentionLevel1MemoryEstimate(
    const GQAWorkspaceAggregate& workspace,
    Level1MemoryEstimate& estimate);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif
