// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "core/common/safeint.h"
#include "core/framework/level1_memory_requirements.h"

namespace onnxruntime::contrib::cuda {

constexpr int64_t kDefaultChunkTargetRows = 32768;
constexpr size_t kMatMulNBitsLegacyScratchLimitBytes = 256 * 1024 * 1024;
constexpr int64_t kMatMulNBitsWeightNotPrepacked = 0;
constexpr int64_t kMatMulNBitsWeightPrepackedSm80 = 1;
constexpr int64_t kMatMulNBitsWeightPrepackedSm90 = 2;
constexpr size_t kMatMulNBitsPermutationMapEntries = 64;

struct MatMulNBitsLegacyWorkspacePlan {
  int64_t k_padded = 0;
  int64_t scratch_rows = 0;
  size_t workspace_bytes = 0;
  bool use_chunked = false;
};

inline std::optional<MatMulNBitsLegacyWorkspacePlan> ComputeMatMulNBitsLegacyWorkspacePlan(
    int64_t n, int64_t k, int64_t block_size, size_t element_size,
    bool can_chunk, bool force_chunked, int64_t chunk_target_rows) {
  if (n < 0 || k < 0 || block_size <= 0 || element_size == 0 ||
      chunk_target_rows <= 0) {
    return std::nullopt;
  }

  int64_t k_rounded = 0;
  int64_t k_padded = 0;
  int64_t minimum_n_for_chunking = 0;
  size_t n_size = 0;
  size_t k_padded_size = 0;
  size_t full_workspace_elements = 0;
  size_t full_workspace_bytes = 0;
  if (!SafeAdd(k, block_size, k_rounded) ||
      !SafeMultiply((k_rounded - 1) / block_size, block_size, k_padded) ||
      !SafeMultiply(chunk_target_rows, 2, minimum_n_for_chunking) ||
      !SafeCast(n, n_size) || !SafeCast(k_padded, k_padded_size) ||
      !SafeMultiply(n_size, k_padded_size, full_workspace_elements) ||
      !SafeMultiply(full_workspace_elements, element_size, full_workspace_bytes)) {
    return std::nullopt;
  }
  const bool use_chunked =
      can_chunk &&
      (force_chunked ||
       (full_workspace_bytes > kMatMulNBitsLegacyScratchLimitBytes &&
        n > minimum_n_for_chunking));
  const int64_t scratch_rows = use_chunked ? chunk_target_rows : n;
  size_t scratch_rows_size = 0;
  size_t workspace_elements = 0;
  size_t workspace_bytes = 0;
  if (!SafeCast(scratch_rows, scratch_rows_size) ||
      !SafeMultiply(scratch_rows_size, k_padded_size, workspace_elements) ||
      !SafeMultiply(workspace_elements, element_size, workspace_bytes)) {
    return std::nullopt;
  }
  return MatMulNBitsLegacyWorkspacePlan{
      k_padded, scratch_rows, workspace_bytes, use_chunked};
}

struct MatMulNBitsPrepackBufferSizes {
  size_t packed_weight_bytes = 0;
  size_t scale_bytes = 0;
  size_t zero_point_bytes = 0;
};

inline std::optional<MatMulNBitsPrepackBufferSizes> ComputeMatMulNBitsPrepackBufferSizes(
    int64_t n, int64_t k, int64_t nbits, int64_t block_size, bool has_zero_points) {
  if (n <= 0 || k <= 0 ||
      (nbits != 2 && nbits != 4 && nbits != 8) ||
      block_size <= 0) {
    return std::nullopt;
  }

  size_t n_size = 0;
  size_t k_size = 0;
  size_t block_size_value = 0;
  size_t weight_elements = 0;
  size_t k_rounded = 0;
  size_t scale_elements = 0;
  size_t scale_bytes = 0;
  if (!SafeCast(n, n_size) || !SafeCast(k, k_size) || !SafeCast(block_size, block_size_value) ||
      !SafeMultiply(n_size, k_size, weight_elements) ||
      !SafeAdd(k_size, block_size_value - 1, k_rounded) ||
      !SafeMultiply(n_size, k_rounded / block_size_value, scale_elements) ||
      !SafeMultiply(scale_elements, sizeof(uint16_t), scale_bytes)) {
    return std::nullopt;
  }
  const size_t packed_weight_bytes = weight_elements / static_cast<size_t>(8 / nbits);
  return MatMulNBitsPrepackBufferSizes{packed_weight_bytes, scale_bytes, has_zero_points ? scale_bytes : 0};
}

// Keep the legacy summary unchanged. The planning breakdown names actual destinations and scratch separately.
inline std::optional<Level1MemoryEstimate> ComputeMatMulNBitsPrepackMemoryEstimate(
    int64_t n, int64_t k, int64_t nbits, int64_t block_size,
    int64_t weight_prepacked, bool has_zero_points) {
  const auto sizes = ComputeMatMulNBitsPrepackBufferSizes(n, k, nbits, block_size, has_zero_points);
  if (!sizes.has_value()) {
    return std::nullopt;
  }

  Level1MemoryEstimate estimate;
  if (!SafeAdd(sizes->scale_bytes, sizes->zero_point_bytes, estimate.persistent_prepack_bytes)) {
    return std::nullopt;
  }
  if (weight_prepacked == kMatMulNBitsWeightNotPrepacked) {
    size_t persistent_prepack_bytes = 0;
    constexpr size_t kLegacyPermutationMapBytes = 32 * sizeof(int32_t);
    if (!SafeAdd(estimate.persistent_prepack_bytes, sizes->packed_weight_bytes, persistent_prepack_bytes) ||
        !SafeAdd(sizes->packed_weight_bytes, kLegacyPermutationMapBytes, estimate.initialization_scratch_bytes)) {
      return std::nullopt;
    }
    estimate.persistent_prepack_bytes = persistent_prepack_bytes;
  }
  return estimate;
}

enum class MatMulNBitsMemoryRoute {
  Unresolved,
  Legacy,
  FpAIntB,
};

struct MatMulNBitsMemoryPlanningFacts {
  std::optional<bool> prepack_inputs_are_constant_initializers;
  // Required only for an offline-prepacked weight. False means PrePack_B needs a new device copy.
  std::optional<bool> offline_weight_on_execution_device;
  // Cached tactic validation can allocate for buckets outside the initial profiling set.
  std::optional<bool> persistent_tactic_cache_load_disabled;
};

// Exact inputs and intermediate sizing results from one estimator invocation, without graph/CUDA types.
struct MatMulNBitsMemorySizingContext final : CapabilityPlanningState {
  CapabilityAllocationProvenance provenance;
  MatMulNBitsMemoryPlanningFacts facts;
  MatMulNBitsMemoryRoute route = MatMulNBitsMemoryRoute::Unresolved;
  std::optional<Level1MemoryEstimate> estimate;
  std::optional<int64_t> m;
  int64_t n = 0;
  int64_t k = 0;
  int64_t bits = 0;
  int64_t block_size = 0;
  int64_t weight_prepacked = 0;
  int32_t input_element_type = 0;
  size_t element_size = 0;
  bool has_zero_points = false;
  bool has_g_idx = false;
  bool has_bias = false;
  bool input_shape_is_upper_bound = false;
  int device_sm = 0;
  int sm_count = 0;
  int l2_cache_bytes = 0;
  bool fpa_intb_build_enabled = false;
  bool compact_fpa_intb_build = false;
  std::string fpa_intb_gemm;
  std::string profile_m;
  std::string m_chunk_size;
  std::string dequant_gemm_min_m;
  bool force_chunked = false;
  int64_t chunk_target_rows = 0;
  std::optional<MatMulNBitsLegacyWorkspacePlan> legacy_workspace;
  std::optional<MatMulNBitsPrepackBufferSizes> prepack_buffers;
  std::optional<size_t> constructor_profile_scratch_bytes;
  std::optional<size_t> cutlass_workspace_bytes;
  std::optional<size_t> lazy_profile_scratch_bytes;
  bool workspace_retained_during_preparation = false;
  std::optional<size_t> dequant_workspace_bytes;
};

inline CapabilityLevel1MemoryEstimate DescribeMatMulNBitsMemoryEstimate(
    const std::shared_ptr<const MatMulNBitsMemorySizingContext>& context) {
  CapabilityLevel1MemoryEstimate source;
  if (!context) {
    source.diagnostic = "MatMulNBits sizing requires a non-null owned sizing context.";
    return source;
  }
  source.provenance = context->provenance;
  source.provenance.sizing_context = context;
  source.scope = Level1MemoryEstimateScope::ImplementationAllocations;
  source.estimate = context->estimate;
  if (!source.estimate.has_value()) {
    source.diagnostic = "MatMulNBits sizing is unavailable for the retained inputs and effective configuration.";
    return source;
  }
  const auto device = source.provenance.query ? source.provenance.query->execution_device : std::nullopt;
  if (!device.has_value() || device->Type() != OrtDevice::GPU ||
      device->Vendor() != OrtDevice::VendorIds::NVIDIA) {
    source.estimate.reset();
    source.diagnostic = "MatMulNBits CUDA sizing requires its NVIDIA execution device.";
    return source;
  }
  const auto allocation = [&](std::string id, size_t bytes) {
    Level1MemoryAllocation result;
    result.resource_id = std::move(id);
    result.device = device;
    result.size_bound_bytes = bytes;
    result.alignment_bytes = 0;
    return result;
  };
  const auto prove_zero = [](Level1MemoryCategoryEvidence& evidence) {
    evidence.allocations.emplace();
  };
  auto& breakdown = source.breakdown;
  if (context->route == MatMulNBitsMemoryRoute::Legacy) {
    if (context->legacy_workspace.has_value()) {
      breakdown.execution_workspace.allocations =
          InlinedVector<Level1MemoryAllocation, 1>{allocation("legacy_dequantized_weight",
                                                              context->legacy_workspace->workspace_bytes)};
    }
    prove_zero(breakdown.persistent_prepack);
    if (!context->has_g_idx) {
      prove_zero(breakdown.initialization_scratch);
      prove_zero(breakdown.runtime_preparation);
    } else {
      breakdown.initialization_scratch.diagnostic = "Group-index validation host scratch is not bounded by L1.";
      breakdown.runtime_preparation.diagnostic = "Runtime group-index validation host scratch is not bounded by L1.";
    }
    return source;
  }
  if (context->route != MatMulNBitsMemoryRoute::FpAIntB) {
    source.diagnostic = "MatMulNBits implementation route is unresolved.";
    return source;
  }

  if (context->dequant_workspace_bytes.has_value()) {
    breakdown.execution_workspace.diagnostic =
        "CUTLASS and dequantize-GEMM alternative allocations lack a validated common backing description.";
  } else if (context->cutlass_workspace_bytes.has_value()) {
    breakdown.execution_workspace.allocations =
        InlinedVector<Level1MemoryAllocation, 1>{allocation("cutlass_workspace", *context->cutlass_workspace_bytes)};
  }

  const bool initializers_verified = context->facts.prepack_inputs_are_constant_initializers.value_or(false);
  if (initializers_verified && context->prepack_buffers.has_value()) {
    const auto& sizes = *context->prepack_buffers;
    auto& persistent = breakdown.persistent_prepack;
    persistent.allocations.emplace();
    if (context->weight_prepacked == kMatMulNBitsWeightNotPrepacked) {
      persistent.allocations->push_back(allocation("packed_weight", sizes.packed_weight_bytes));
    } else if (!context->facts.offline_weight_on_execution_device.has_value()) {
      persistent.diagnostic = "Offline-prepacked weight residency is unknown; reuse in place is not proved.";
    } else if (!*context->facts.offline_weight_on_execution_device) {
      persistent.allocations->push_back(allocation("offline_weight_device_copy", sizes.packed_weight_bytes));
    }
    persistent.allocations->push_back(allocation("packed_scales", sizes.scale_bytes));
    if (context->has_zero_points) {
      persistent.allocations->push_back(allocation("scaled_zero_points", sizes.zero_point_bytes));
    }
  } else {
    breakdown.persistent_prepack.diagnostic = "Prepack destinations require verified constant inputs and buffer sizes.";
  }

  auto& initialization = breakdown.initialization_scratch;
  initialization.allocations.emplace();
  if (context->constructor_profile_scratch_bytes.has_value()) {
    if (*context->constructor_profile_scratch_bytes > 0) {
      initialization.allocations->push_back(allocation("constructor_profiler", *context->constructor_profile_scratch_bytes));
    }
  } else {
    initialization.diagnostic = "Constructor profiling allocation coverage is unresolved.";
  }
  if (!context->facts.persistent_tactic_cache_load_disabled.value_or(false)) {
    initialization.diagnostic +=
        " Persistent tactic-cache validation can allocate beyond the L1 initial-profile peak; cache loading is not proved disabled.";
  }
  if (context->weight_prepacked == kMatMulNBitsWeightNotPrepacked) {
    if (initializers_verified && context->prepack_buffers.has_value()) {
      initialization.allocations->push_back(allocation("prepack_transposed_weight",
                                                       context->prepack_buffers->packed_weight_bytes));
      initialization.allocations->push_back(allocation("prepack_permutation_map",
                                                       kMatMulNBitsPermutationMapEntries * sizeof(int32_t)));
    } else {
      initialization.diagnostic += " Prepack conversion allocation coverage is unresolved.";
    }
  }

  if (context->lazy_profile_scratch_bytes.has_value()) {
    auto& preparation = breakdown.runtime_preparation;
    preparation.allocations.emplace();
    if (*context->lazy_profile_scratch_bytes > 0) {
      preparation.allocations->push_back(allocation("lazy_profiler", *context->lazy_profile_scratch_bytes));
    }
    breakdown.runtime_preparation_allocations_exclude_workspace = true;
  } else {
    breakdown.runtime_preparation.diagnostic = "Lazy profiling allocation coverage is unresolved.";
  }
  return source;
}

}  // namespace onnxruntime::contrib::cuda
