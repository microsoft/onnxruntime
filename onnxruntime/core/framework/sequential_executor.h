// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/iexecutor.h"
#include "core/framework/framework_common.h"
#include "core/framework/ort_value.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/common/inlined_containers.h"

#ifdef ENABLE_TRAINING
#include "core/framework/partial_graph_execution_state.h"
#endif

namespace onnxruntime {

class ExecutionContext;
class DeviceStreamCollection;
typedef InlinedHashMap<std::string, OrtValue> OrtValueCache;
typedef std::shared_ptr<OrtValueCache> OrtValueCachePtr;

onnxruntime::Status ExecuteKernel(ExecutionContext& ctx, NodeIndex idx, size_t stream_idx);

onnxruntime::Status ExecuteThePlan(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                                   gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                   std::vector<OrtValue>& fetches,
                                   const InlinedHashMap<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                   const logging::Logger& logger,
                                   const DeviceStreamCollection& device_streams,
                                   const bool* terminate_flag,
                                   const bool only_execute_path_to_fetches,
                                   bool single_thread_mode);

#ifdef ENABLE_TRAINING
onnxruntime::Status PartialExecuteThePlan(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                                          gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                          std::vector<OrtValue>& fetches,
                                          const InlinedHashMap<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                          const logging::Logger& logger,
                                          const DeviceStreamCollection& device_streams,
                                          const bool* terminate_flag,
                                          bool single_thread_mode,
                                          PartialGraphExecutionState& state,
                                          const OrtValueCachePtr& cache);
#endif
}  // namespace onnxruntime
