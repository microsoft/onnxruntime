// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/common/scatter_mean.h"
#include "core/platform/threadpool.h"

namespace onnxruntime::scatter_internal {

template <typename T>
Status ScatterMean(const Tensor& data, const Tensor& updates, gsl::span<const int64_t> destinations,
                   int64_t slice_size, concurrency::ThreadPool* tp, Tensor& output) {
  if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, std::string> ||
                (!std::is_arithmetic_v<T> && !std::is_same_v<T, MLFloat16> && !std::is_same_v<T, BFloat16>)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Scatter reduction 'mean' requires integer or floating-point data; cast data and updates "
                           "to a supported numeric type.");
  } else {
    const int64_t size = data.Shape().Size();
    if (size == 0) return Status::OK();
    const auto* input = data.Data<T>();
    auto* result = output.MutableData<T>();
    if (input != result) {
      memcpy(result, input, data.SizeInBytes());
    }
    if (destinations.empty() || slice_size == 0) return Status::OK();

    InlinedVector<int64_t> heads(static_cast<size_t>(size / slice_size), -1);
    InlinedVector<int64_t> next(destinations.size(), -1);
    // Build in reverse so each destination consumes updates in their input order.
    for (size_t i = destinations.size(); i-- > 0;) {
      auto& head = heads[static_cast<size_t>(destinations[i] / slice_size)];
      next[i] = head;
      head = static_cast<int64_t>(i);
    }
    const auto* update_data = updates.Data<T>();
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<size_t>(size), 1.0 + static_cast<double>(updates.Shape().Size()) / size,
        [&](ptrdiff_t first, ptrdiff_t last) {
          for (ptrdiff_t i = first; i < last; ++i) {
            const int64_t head = heads[static_cast<size_t>(i / slice_size)];
            if (head >= 0) {
              result[i] = MeanOfLinkedUpdates(input[i], update_data, next.data(), head, slice_size, i % slice_size);
            }
          }
        });
    return Status::OK();
  }
}

}  // namespace onnxruntime::scatter_internal
