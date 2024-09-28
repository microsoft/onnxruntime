// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/profiler_common.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {

namespace webgpu {
class WebGpuContext;
}

namespace profiling {

class WebGPUProfiler final : public EpProfiler {
 public:
  WebGPUProfiler(int context_id);
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGPUProfiler);
  ~WebGPUProfiler() {}
  bool StartProfiling(TimePoint) override;
  void EndProfiling(TimePoint, Events&) override;
  void Start(uint64_t) override{};
  void Stop(uint64_t) override{};

 private:
  webgpu::WebGpuContext& webgpu_context_;
};

}  // namespace profiling
}  // namespace onnxruntime
