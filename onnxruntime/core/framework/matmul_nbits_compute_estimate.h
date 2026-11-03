// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <gsl/span>

namespace onnxruntime {
class Node;
struct MatMulNBitsRuntimeMetadata;

namespace contrib {

enum class ComputeEstimateUnavailableReason {
  UnsupportedOperator,
  UnsupportedVersion,
  UnknownShape,
  InvalidMetadata,
  Overflow,
  MissingSemanticValue,
  UnsupportedRoute,
};

struct MatMulNBitsComputeEstimate {
  // Convention v1: two operations per multiply-add, counting only the logical matrix product.
  // Excludes bias, dequantization, packing, conversions, padding, and route-specific extra work.
  static constexpr uint32_t kConventionVersion = 1;
  uint64_t dense_equivalent_ops;
};

using MatMulNBitsComputeEstimateResult =
    std::variant<MatMulNBitsComputeEstimate, ComputeEstimateUnavailableReason>;

struct MatMulNBitsBiasEstimate {
  uint64_t floating_point_ops;
};

using MatMulNBitsBiasEstimateResult =
    std::variant<MatMulNBitsBiasEstimate, ComputeEstimateUnavailableReason>;

// A has shape [..., K]. Rank-one inputs have M = 1. -1 denotes an unknown dimension;
// unknown dimensions remain unavailable even when another dimension is zero.
// This is a metadata-only estimate, not validation of packed weights or other operator inputs.
MatMulNBitsComputeEstimateResult EstimateMatMulNBitsCompute(
    gsl::span<const int64_t> input_a_shape, int64_t k, int64_t n);

// Counts MN logical additions, not route-specific executed work. Known-absent bias is zero
// without requiring shape metadata; unknown presence is unavailable. Present bias uses
// the same shape-validity rules as the core, but K = 0 does not imply empty output.
MatMulNBitsBiasEstimateResult EstimateMatMulNBitsBias(
    gsl::span<const int64_t> input_a_shape, int64_t k, int64_t n, std::optional<bool> bias_present);

#if !defined(ORT_MINIMAL_BUILD)
enum class MatMulNBitsCpuFlopsModel {
  UnpackedFloat,
  PackedFloat,
  PackedInt8,
  NotExecuted,
};

struct MatMulNBitsCpuFlopsEstimate {
  std::optional<MatMulNBitsCpuFlopsModel> model;
  MatMulNBitsBiasEstimateResult core;
  MatMulNBitsBiasEstimateResult bias;
  MatMulNBitsBiasEstimateResult dequantization;
};

// Ballpark v1: floating core 2MNK; bias MN; one nominal floating reconstruction,
// 2NK unpacked or 2N*ceil(K/block_size)*block_size packed. Integer core is zero FLOPs;
// nominal block scaling/correction is 5MN*ceil(K/block_size). Excludes activation
// quantization, conversions, preparation, auxiliary reductions and worker/tile repetition.
MatMulNBitsCpuFlopsEstimate EstimateMatMulNBitsCpuFlops(const MatMulNBitsRuntimeMetadata& metadata);

// Reads canonical metadata from a resolved in-tree graph, never maximum-shape hints.
// The same core count applies across providers, dtypes, and supported quantization bit widths.
MatMulNBitsComputeEstimateResult EstimateMatMulNBitsCompute(const Node& node);
MatMulNBitsBiasEstimateResult EstimateMatMulNBitsBias(const Node& node);
#endif

}  // namespace contrib
}  // namespace onnxruntime
