// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

// TLS definition for run-level profiling
thread_local WebGpuProfiler::RunLevelState WebGpuProfiler::run_level_state_;

WebGpuProfiler::WebGpuProfiler(WebGpuContext& context) : context_{context} {}

bool WebGpuProfiler::StartProfiling(TimePoint) {
  enabled_ = true;
  return true;
}

void WebGpuProfiler::EndProfiling(TimePoint tp, onnxruntime::profiling::Events& events) {
  context_.EndProfiling(tp, events, events_);
  enabled_ = false;
}

void WebGpuProfiler::StartRunProfiling() {
  run_level_state_.enabled = true;
  run_level_state_.events.clear();
}

void WebGpuProfiler::EndRunProfiling(TimePoint /* start_time */, onnxruntime::profiling::Events& events) {
  if (!run_level_state_.enabled) {
    return;
  }

  // Move collected GPU events to the output
  for (auto& event : run_level_state_.events) {
    events.emplace_back(std::move(event));
  }

  run_level_state_.events.clear();
  run_level_state_.enabled = false;
}

bool WebGpuProfiler::IsRunProfilingEnabled() const {
  return run_level_state_.enabled;
}

void WebGpuProfiler::DispatchGpuEvents(onnxruntime::profiling::Events& gpu_events) {
  // If session-level profiling is enabled, copy to session events
  if (enabled_) {
    for (const auto& event : gpu_events) {
      events_.emplace_back(event);  // copy
    }
  }

  // If run-level profiling is enabled, move to run-level events
  if (run_level_state_.enabled) {
    for (auto& event : gpu_events) {
      run_level_state_.events.emplace_back(std::move(event));  // move
    }
  }
}

}  // namespace webgpu
}  // namespace onnxruntime
