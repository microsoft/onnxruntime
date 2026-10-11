// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iterator>

#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

WebGpuProfiler::WebGpuProfiler(WebGpuContext& context) : context_{context} {}

Status WebGpuProfiler::StartProfiling(TimePoint profiling_start_time) {
  enabled_ = true;
  // Push the ORT profiler's CPU time base into the context so GPU timestamps align
  // with ORT CPU events. This is the only hook that receives the framework's
  // profiling_start_time for both session-level and run-level profiling; for run-level
  // profiling the WebGpuProfiler is a temporary that OnRunStart cannot reach directly.
  context_.SetProfilingStartTime(profiling_start_time);
  return Status::OK();
}

void WebGpuProfiler::EndProfiling(TimePoint tp, onnxruntime::profiling::Events& events) {
  if (is_session_level_) {
    // Session-level profiling: drain profiler's own GPU events.
    events.insert(events.end(),
                  std::make_move_iterator(gpu_events_.begin()),
                  std::make_move_iterator(gpu_events_.end()));
    gpu_events_.clear();
  } else {
    // Run-level profiling: drain shared events from the context.
    context_.EndProfiling(tp, events);
  }
  enabled_ = false;
}

}  // namespace webgpu
}  // namespace onnxruntime
