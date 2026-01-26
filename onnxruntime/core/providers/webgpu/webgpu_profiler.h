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
  inline bool Enabled() const { return enabled_; }

 private:
  WebGpuContext& context_;
  bool enabled_{false};
};

}  // namespace webgpu
}  // namespace onnxruntime
