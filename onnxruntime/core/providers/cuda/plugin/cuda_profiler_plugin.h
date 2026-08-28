// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(ENABLE_CUDA_PROFILING)

#include "cuda_plugin_utils.h"
#include "core/providers/cuda/cupti_manager.h"
#include "core/common/gpu_profiler_common.h"

namespace onnxruntime {
namespace cuda_plugin {

/// Plugin-side implementation of OrtEpProfilerImpl for CUDA.
/// Delegates to CUPTIManager (within the plugin DLL) for GPU activity tracing
/// and implements the C callback interface expected by ORT's PluginEpProfiler bridge.
struct CudaPluginEpProfiler : OrtEpProfilerImpl {
  const OrtEpApi& ep_api;
  uint64_t client_handle_ = 0;
  TimePoint ort_profiling_start_;

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
