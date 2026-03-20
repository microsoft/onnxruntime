// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

WebGpuProfiler::WebGpuProfiler(WebGpuContext& context) : context_{context} {}

bool WebGpuProfiler::StartProfiling(TimePoint) {
  enabled_ = true;
  return true;
}

void WebGpuProfiler::EndProfiling(TimePoint tp, onnxruntime::profiling::Events& events) {
  // Drain session-level GPU events (collected via CollectProfilingData(gpu_events_)).
  if (!gpu_events_.empty()) {
    events.insert(events.end(),
                  std::make_move_iterator(gpu_events_.begin()),
                  std::make_move_iterator(gpu_events_.end()));
    gpu_events_.clear();
  }
  // Drain any shared run-level events from the context.
  context_.EndProfiling(tp, events);
  enabled_ = false;
}

}  // namespace webgpu
}  // namespace onnxruntime
