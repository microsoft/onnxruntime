// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/profiler_common.h"

namespace onnxruntime {

namespace webgpu {
class WebGpuContext;

class WebGpuProfiler final : public onnxruntime::profiling::EpProfiler {
 public:
  WebGpuProfiler(WebGpuContext& context);
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuProfiler);
  ~WebGpuProfiler() {}
  bool StartProfiling(TimePoint) override;
  void EndProfiling(TimePoint, onnxruntime::profiling::Events&) override;
  void Start(uint64_t) override {
  }
  void Stop(uint64_t) override {
  }

  // Run-level profiling support
  void StartRunProfiling() override;
  void EndRunProfiling(TimePoint start_time, onnxruntime::profiling::Events& events) override;
  bool IsRunProfilingEnabled() const override;

  // Check if session-level profiling is enabled (original semantics)
  inline bool Enabled() const { return enabled_; }

  // Get session-level events container
  inline onnxruntime::profiling::Events& Events() { return events_; }

  // Dispatch GPU events to appropriate containers based on profiling mode
  // Call this when GPU events are collected
  void DispatchGpuEvents(onnxruntime::profiling::Events& gpu_events);

 private:
  WebGpuContext& context_;
  bool enabled_{false};
  onnxruntime::profiling::Events events_;  // cached GPU events for session-level

  // TLS for run-level profiling
  struct RunLevelState {
    bool enabled = false;
    onnxruntime::profiling::Events events;
  };
  static thread_local RunLevelState run_level_state_;
};

}  // namespace webgpu
}  // namespace onnxruntime
