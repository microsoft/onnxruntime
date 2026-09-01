// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <type_traits>

#include "core/framework/op_kernel_context_internal.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace contrib {

inline const RunInstrumentationContext* GetMoeRunInstrumentationContext(const OpKernelContext* context) {
  return context->GetRunInstrumentationContext();
}

template <typename T>
std::string MoeJsonArray(gsl::span<const T> values) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      stream << ",";
    }
    if constexpr (std::is_floating_point_v<T>) {
      if (std::isfinite(values[i])) {
        stream << std::setprecision(std::numeric_limits<T>::max_digits10) << values[i];
      } else {
        stream << "null";
      }
    } else {
      stream << values[i];
    }
  }
  stream << "]";
  return stream.str();
}

inline void RecordMoeRoutingEvent(const RunInstrumentationContext& instrumentation,
                                  const Node& node,
                                  gsl::span<const int> expert_ids,
                                  gsl::span<const float> router_weights,
                                  int64_t num_rows,
                                  int64_t top_k,
                                  const TimePoint& start_time) {
  const auto completion_time = std::chrono::high_resolution_clock::now();
  const auto completion_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 completion_time.time_since_epoch())
                                 .count() -
                             static_cast<int64_t>(instrumentation.ProfilerStartTimeNs());

  instrumentation.RecordMoeRoutingEvent(
      start_time, completion_time, node.Name(), node.Index(),
      MoeJsonArray(expert_ids), MoeJsonArray(router_weights),
      num_rows, top_k, -1, completion_ns, "host_clock");
}

}  // namespace contrib
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
