// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef ENABLE_TRAINING
#pragma once

#include <vector>
#include <unordered_map>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/iexecutor.h"
#include "core/framework/framework_common.h"
#include "core/framework/ort_value.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/partial_graph_execution_state.h"

namespace onnxruntime {
class PartialExecutor : public IExecutor {
 public:
  PartialExecutor(PartialGraphExecutionState& state,
                  const OrtValueCachePtr& cache,
                  int32_t partial_graph_index = 0)
      : state_{state},
        cache_{cache},
        partial_graph_index_{partial_graph_index} {
    ORT_UNUSED_PARAMETER(partial_graph_index_);
  }

  common::Status Execute(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                         gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger) override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PartialExecutor);
  PartialGraphExecutionState& state_;
  const OrtValueCachePtr& cache_;
  int32_t partial_graph_index_{0};
};
}  // namespace onnxruntime
#endif
