// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

WebGpuProfiler::WebGpuProfiler(WebGpuContext& context) : context_{context} {}

bool WebGpuProfiler::StartProfiling(TimePoint) {
  context_.RegisterProfiler(this);
  enabled_ = true;
  return true;
}

void WebGpuProfiler::EndProfiling(TimePoint tp, onnxruntime::profiling::Events& events) {
  context_.UnregisterProfiler(this);
  events.insert(events.end(), std::make_move_iterator(events_.begin()), std::make_move_iterator(events_.end()));
  events_.clear();
  enabled_ = false;
}

}  // namespace webgpu
}  // namespace onnxruntime
