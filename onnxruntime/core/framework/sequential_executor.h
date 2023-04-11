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

class StreamExecutionContext;
class DeviceStreamCollection;
class SessionScope;

#ifdef ENABLE_TRAINING
using OrtValueCache = InlinedHashMap<std::string, OrtValue>;
using OrtValueCachePtr = std::shared_ptr<OrtValueCache>;
#endif

onnxruntime::Status ExecuteKernel(StreamExecutionContext& ctx,
                                  NodeIndex idx,
                                  size_t stream_idx,
                                  const bool& terminate_flag,
                                  SessionScope& session_scope);

onnxruntime::Status ExecuteThePlan(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                                   gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                   std::vector<OrtValue>& fetches,
                                   const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                   const logging::Logger& logger,
#ifdef ORT_ENABLE_STREAM
                                   const DeviceStreamCollection* device_streams,
#endif
                                   const bool& terminate_flag,
                                   const bool only_execute_path_to_fetches,
                                   bool single_thread_mode);

#ifdef ENABLE_TRAINING
onnxruntime::Status PartialExecuteThePlan(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                                          gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                          std::vector<OrtValue>& fetches,
                                          const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                          const logging::Logger& logger,
                                          const DeviceStreamCollection* device_streams,
                                          const bool& terminate_flag,
                                          bool single_thread_mode,
                                          PartialGraphExecutionState& state,
                                          const OrtValueCachePtr& cache,
                                          int32_t partial_graph_index);
#endif
}  // namespace onnxruntime
