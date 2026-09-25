// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <memory>
#include <vector>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/ort_value.h"
#include "core/framework/run_options.h"

namespace onnxruntime {
class FeedsFetchesManager;
class IExecutionProvider;
class SessionState;

// Experimental, sequential CPU/CUDA execution with a captured graph per CUDA partition.
class PartitionedGraphExecution {
 public:
  PartitionedGraphExecution(const SessionState& session_state, IExecutionProvider& provider);
  ~PartitionedGraphExecution();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PartitionedGraphExecution);

  Status Run(const RunOptions& run_options, int graph_id,
             FeedsFetchesManager& feeds_fetches_manager,
             gsl::span<const OrtValue> feeds, std::vector<OrtValue>& fetches,
             const logging::Logger& logger);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
