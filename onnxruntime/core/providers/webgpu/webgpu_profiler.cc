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
  context_.EndProfiling(tp, events, events_);
  enabled_ = false;
}

}  // namespace webgpu
}  // namespace onnxruntime
