// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <optional>

#include <gsl/span>

#include "core/common/common.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace contrib {
namespace attention {

constexpr int64_t kMaxAttentionDimension = (std::numeric_limits<int>::max)();

inline Status CheckAttentionWeights(gsl::span<const int64_t> weights_dims,
                                    gsl::span<const int64_t> qkv_hidden_sizes,
                                    int num_heads,
                                    bool require_same_hidden_size,
                                    std::array<int64_t, 3>& hidden_sizes) {
  ORT_RETURN_IF_NOT(weights_dims.size() == 2, "Input 'weights' is expected to have 2 dimensions");
  ORT_RETURN_IF_NOT(num_heads > 0, "num_heads must be positive");
  for (int64_t dim : weights_dims) {
    ORT_RETURN_IF_NOT(dim >= 0 && dim <= kMaxAttentionDimension,
                      "Input 'weights' dimensions must be in [0, INT_MAX]");
  }

  ORT_RETURN_IF_NOT(qkv_hidden_sizes.empty() || qkv_hidden_sizes.size() == 3,
                    "qkv_hidden_sizes attribute should have 3 elements");
  if (qkv_hidden_sizes.empty() || require_same_hidden_size) {
    ORT_RETURN_IF_NOT(weights_dims[1] % 3 == 0,
                      "Input 'weights' dimension 1 must be a multiple of 3 for equal Q/K/V hidden sizes");
  }

  std::array<int64_t, 3> sizes{weights_dims[1] / 3, weights_dims[1] / 3, weights_dims[1] / 3};
  if (!qkv_hidden_sizes.empty()) {
    std::copy(qkv_hidden_sizes.begin(), qkv_hidden_sizes.end(), sizes.begin());
  }

  for (int64_t size : sizes) {
    ORT_RETURN_IF_NOT(size > 0 && size <= kMaxAttentionDimension,
                      "Q, K and V hidden sizes must be in [1, INT_MAX]");
    ORT_RETURN_IF_NOT(size % num_heads == 0, "hidden_size should be divisible by num_heads:", size);
  }
  ORT_RETURN_IF_NOT(sizes[0] == sizes[1], "qkv_hidden_sizes first element should be same as the second");
  ORT_RETURN_IF_NOT(!require_same_hidden_size || sizes[1] == sizes[2], "Hidden size of Q, K and V shall be same");

  const int64_t total_hidden_size = SafeInt<int64_t>(sizes[0]) + sizes[1] + sizes[2];
  ORT_RETURN_IF_NOT(total_hidden_size == weights_dims[1],
                    "Input 'weights' dimension 1 should have same length as sum of Q/K/V hidden sizes");
  hidden_sizes = sizes;
  return Status::OK();
}

inline Status CheckAttentionProjectionSize(int64_t batch_size, int64_t sequence_length, int64_t projection_width) {
  ORT_RETURN_IF_NOT(batch_size >= 0 && batch_size <= kMaxAttentionDimension &&
                        sequence_length >= 0 && sequence_length <= kMaxAttentionDimension &&
                        projection_width > 0 && projection_width <= kMaxAttentionDimension,
                    "Attention projection dimensions are outside the supported range");

  // Projection transposes and per-head offsets use int indices.
  int64_t elements = 0;
  size_t size = 0;
  ORT_RETURN_IF_NOT(SafeMultiply(batch_size, sequence_length, elements) &&
                        SafeMultiply(elements, projection_width, elements) &&
                        elements <= kMaxAttentionDimension && SafeCast(elements, size) &&
                        size <= (std::numeric_limits<size_t>::max)() / sizeof(float),
                    "Attention projection size exceeds the supported range");
  return Status::OK();
}

inline Status CheckAttentionSequenceLengths(int64_t sequence_length,
                                            int64_t past_sequence_length,
                                            std::optional<int64_t> cache_capacity,
                                            int64_t& total_sequence_length) {
  ORT_RETURN_IF_NOT(sequence_length >= 0 && sequence_length <= kMaxAttentionDimension,
                    "sequence_length must be in [0, INT_MAX]");
  ORT_RETURN_IF_NOT(past_sequence_length >= 0 && past_sequence_length <= kMaxAttentionDimension,
                    "past_sequence_length must be in [0, INT_MAX]");
  if (cache_capacity.has_value()) {
    ORT_RETURN_IF_NOT(*cache_capacity >= 0 && *cache_capacity <= kMaxAttentionDimension,
                      "past cache capacity must be in [0, INT_MAX]");
    ORT_RETURN_IF_NOT(past_sequence_length <= *cache_capacity &&
                          sequence_length <= *cache_capacity - past_sequence_length,
                      "past_sequence_length plus sequence_length must not exceed past cache capacity");
  }
  ORT_RETURN_IF_NOT(sequence_length <= kMaxAttentionDimension - past_sequence_length,
                    "total_sequence_length must not exceed INT_MAX");
  total_sequence_length = SafeInt<int64_t>(past_sequence_length) + sequence_length;
  return Status::OK();
}

}  // namespace attention
}  // namespace contrib
}  // namespace onnxruntime
