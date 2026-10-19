// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantization/matmul_nbits_compute_estimate.h"

#include <algorithm>

#include "core/common/safeint.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/common/inlined_containers.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#endif

namespace onnxruntime {
namespace contrib {

MatMulNBitsComputeEstimateResult EstimateMatMulNBitsCompute(
    gsl::span<const int64_t> input_a_shape, int64_t k, int64_t n) {
  if (input_a_shape.empty() || k < 0 || n < 0 ||
      std::any_of(input_a_shape.begin(), input_a_shape.end(), [](int64_t dim) { return dim < -1; })) {
    return ComputeEstimateUnavailableReason::InvalidMetadata;
  }

  if (input_a_shape.back() >= 0 && input_a_shape.back() != k) {
    return ComputeEstimateUnavailableReason::InvalidMetadata;
  }

  if (std::find(input_a_shape.begin(), input_a_shape.end(), -1) != input_a_shape.end()) {
    return ComputeEstimateUnavailableReason::UnknownShape;
  }

  if (n == 0 || std::find(input_a_shape.begin(), input_a_shape.end(), 0) != input_a_shape.end()) {
    return MatMulNBitsComputeEstimate{0};
  }

  uint64_t count = 2;
  if (!SafeMultiply(count, static_cast<uint64_t>(n), count)) {
    return ComputeEstimateUnavailableReason::Overflow;
  }

  // Multiplying every A dimension incorporates both the leading-dimension product M and K.
  for (int64_t dim : input_a_shape) {
    if (!SafeMultiply(count, static_cast<uint64_t>(dim), count)) {
      return ComputeEstimateUnavailableReason::Overflow;
    }
  }

  return MatMulNBitsComputeEstimate{count};
}

#if !defined(ORT_MINIMAL_BUILD)
MatMulNBitsComputeEstimateResult EstimateMatMulNBitsCompute(const Node& node) {
  if (node.Domain() != kMSDomain || node.OpType() != "MatMulNBits") {
    return ComputeEstimateUnavailableReason::UnsupportedOperator;
  }

  if (node.SinceVersion() != 1) {
    return ComputeEstimateUnavailableReason::UnsupportedVersion;
  }

  const auto& attributes = node.GetAttributes();
  const auto k_it = attributes.find("K");
  const auto n_it = attributes.find("N");
  if (k_it == attributes.end() || n_it == attributes.end() ||
      k_it->second.type() != ONNX_NAMESPACE::AttributeProto::INT || !k_it->second.has_i() ||
      n_it->second.type() != ONNX_NAMESPACE::AttributeProto::INT || !n_it->second.has_i() ||
      k_it->second.i() < 0 || n_it->second.i() < 0) {
    return ComputeEstimateUnavailableReason::InvalidMetadata;
  }

  const auto& inputs = node.InputDefs();
  if (inputs.empty() || inputs[0] == nullptr || !inputs[0]->Exists()) {
    return ComputeEstimateUnavailableReason::InvalidMetadata;
  }

  const auto* shape = inputs[0]->Shape();
  if (shape == nullptr) {
    return ComputeEstimateUnavailableReason::UnknownShape;
  }

  InlinedVector<int64_t> dimensions;
  dimensions.reserve(shape->dim_size());
  for (const auto& dim : shape->dim()) {
    if (dim.has_dim_value()) {
      if (dim.dim_value() < 0) {
        return ComputeEstimateUnavailableReason::InvalidMetadata;
      }
      dimensions.push_back(dim.dim_value());
    } else {
      dimensions.push_back(-1);
    }
  }

  return EstimateMatMulNBitsCompute(dimensions, k_it->second.i(), n_it->second.i());
}
#endif

}  // namespace contrib
}  // namespace onnxruntime
