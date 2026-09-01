// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <cuda_runtime_api.h>
#include <gsl/span>

#include "core/common/inlined_containers.h"
#include "core/framework/workspace_input_shape.h"
#include "core/framework/workspace_requirement.h"
#include "contrib_ops/cuda/bert/attention_kernel_options.h"
#include "contrib_ops/cuda/bert/packed_attention_workspace.h"

namespace onnxruntime {
// Do not forward-declare Node here. This adapter is included from both the
// in-tree graph world and the shared-provider bridge world, which define Node
// with different class keys. Each includer must provide its own declaration.
namespace contrib {
namespace cuda {

enum class PackedAttentionWorkspaceOperator {
  PackedAttention,
  PackedMultiHeadAttention,
};

struct PackedAttentionWorkspaceEstimateConfig {
  PackedAttentionWorkspaceOperator op = PackedAttentionWorkspaceOperator::PackedAttention;
  size_t element_size = 0;
  int64_t num_heads = 0;
  size_t qkv_hidden_sizes_count = 0;
  std::array<int64_t, 3> qkv_hidden_sizes{};
};

// The problem geometry is an upper-bound hint. Returned routes are those that
// may be reachable at some valid runtime geometry up to that bound.
PackedAttentionBackendMask GetPackedAttentionFeasibleBackends(
    const PackedAttentionProblem& problem,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options);

PackedAttentionBackendMask GetPackedMultiHeadAttentionFeasibleBackends(
    const PackedMultiHeadAttentionProblem& problem,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options);

// Translates positional framework shapes into the graph-free problem and then
// aggregates every route potentially reachable up to the supplied geometry.
// nullopt means required metadata, route feasibility, or checked recipe
// arithmetic was unavailable.
std::optional<PackedAttentionWorkspaceAggregate> EstimatePackedAttentionWorkspace(
    const PackedAttentionWorkspaceEstimateConfig& config,
    gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options);

// Level-1 wrapper. Node provides the operator type, attributes, and input-0
// dtype. Positional WorkspaceInputShape entries provide optional-input
// presence and geometry.
std::optional<PackedAttentionWorkspaceAggregate> EstimatePackedAttentionWorkspace(
    const Node& node,
    gsl::span<const WorkspaceInputShape> input_shapes,
    const cudaDeviceProp& device_prop,
    const AttentionKernelOptions& kernel_options);

// Converts a successful nonzero aggregate to one explicitly aligned Level-2
// root requirement. Failed and zero-byte aggregates emit no requirement.
void SetPackedAttentionWorkspaceRequirements(
    const PackedAttentionWorkspaceAggregate& estimate,
    InlinedVector<WorkspaceRequirement>& requirements);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)
