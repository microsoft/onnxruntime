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
  Status StartProfiling(TimePoint) override;
  void EndProfiling(TimePoint, onnxruntime::profiling::Events&) override;
  void Start(uint64_t) override {
  }
  void Stop(uint64_t, const profiling::EventRecord&) override {
  }
  inline bool Enabled() const { return enabled_; }
  // GPU events collected during session-level profiling.
  profiling::Events& GpuEvents() {
    is_session_level_ = true;
    return gpu_events_;
  }

 private:
  WebGpuContext& context_;
  bool enabled_{false};
  bool is_session_level_{false};
  profiling::Events gpu_events_;
};

}  // namespace webgpu
}  // namespace onnxruntime
