// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(ENABLE_CUDA_PROFILING)

#include <mutex>
#include <string>
#include <unordered_map>

#include "cuda_plugin_utils.h"
#include "core/providers/cuda/cupti_manager.h"
#include "core/common/gpu_profiler_common.h"

namespace onnxruntime {
namespace cuda_plugin {

/// Per-node ORT profiling metadata captured during StopEvent and used in
/// EndProfiling to annotate CUPTI-captured GPU events with explicit
/// ORT-side attribution (node name, op type, node index).
struct OrtNodeInfo {
  std::string event_name;  ///< Full ORT event name (e.g. "<node>_kernel_time").
  std::string op_name;     ///< ONNX op type for the node, if available.
  std::string node_index;  ///< Node index in the graph as a decimal string, if available.
};

/// Plugin-side implementation of OrtEpProfilerImpl for CUDA.
/// Delegates to CUPTIManager (within the plugin DLL) for GPU activity tracing
/// and implements the C callback interface expected by ORT's PluginEpProfiler bridge.
struct CudaPluginEpProfiler : OrtEpProfilerImpl {
  const OrtEpApi& ep_api;
  uint64_t client_handle_ = 0;
  TimePoint ort_profiling_start_;

  // Maps the absolute, epoch-based ORT event correlation ID for a NODE_EVENT
  // (as passed to StartEvent/StopEvent) to the originating node's identity.
  // Populated in StopEventImpl and drained in EndProfilingImpl, where the
  // entries are joined against CUPTI-captured GPU events to attribute each
  // GPU kernel back to a specific ORT graph node.
  //
  // Different ORT events may run on different threads (inter-op parallelism),
  // so map access is protected by node_info_mutex_.
  std::mutex node_info_mutex_;
  std::unordered_map<uint64_t, OrtNodeInfo> correlation_to_node_;

  explicit CudaPluginEpProfiler(const OrtEpApi& api);
  ~CudaPluginEpProfiler();

  static void ORT_API_CALL ReleaseImpl(OrtEpProfilerImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL StartProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                    int64_t ep_profiling_start_offset_ns) noexcept;
  static OrtStatus* ORT_API_CALL StartEventImpl(OrtEpProfilerImpl* this_ptr,
                                                uint64_t ort_event_correlation_id) noexcept;
  static OrtStatus* ORT_API_CALL StopEventImpl(OrtEpProfilerImpl* this_ptr,
                                               uint64_t ort_event_correlation_id,
                                               const OrtProfilingEvent* ort_event) noexcept;
  static OrtStatus* ORT_API_CALL EndProfilingImpl(OrtEpProfilerImpl* this_ptr,
                                                  OrtProfilingEventsContainer* events_container) noexcept;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime

#endif  // defined(ENABLE_CUDA_PROFILING)
