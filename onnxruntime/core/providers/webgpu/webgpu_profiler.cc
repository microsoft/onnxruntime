// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iterator>

#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

WebGpuProfiler::WebGpuProfiler(WebGpuContext& context) : context_{context} {}

Status WebGpuProfiler::StartProfiling(TimePoint) {
  enabled_ = true;
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
