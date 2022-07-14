// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

#ifdef ENABLE_TRAINING
#include "core/framework/partial_graph_execution_state.h"
#endif 

namespace onnxruntime {

class ExecutionContext;

onnxruntime::Status BindToDeviceStream(Stream* parent_stream,
                                  const SequentialExecutionPlan& execution_plan,
                                  DeviceStreamColloection& device_stream_map,
                                  IStreamCommandHandleRegistry& stream_handle_registry);

onnxruntime::Status ExecuteKernel(ExecutionContext& ctx, NodeIndex idx, size_t stream_idx);

onnxruntime::Status ExecuteThePlan(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                      const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                      std::vector<OrtValue>& fetches,
                                      const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                      const logging::Logger& logger,
                                      const DeviceStreamColloection& device_streams,
                                      const bool& terminate_flag,
                                      const bool only_execute_path_to_fetches,
                                      bool single_thread_mode);

#ifdef ENABLE_TRAINING
onnxruntime::Status PartialExecuteThePlan(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                                          const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                                          std::vector<OrtValue>& fetches,
                                          const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                                          const logging::Logger& logger,
                                          const DeviceStreamColloection& device_streams,
                                          const bool& terminate_flag,
                                          bool single_thread_mode,
                                          PartialGraphExecutionState& state,
                                          const OrtValueCachePtr& cache);
#endif
}  // namespace onnxruntime
