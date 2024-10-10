// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace profiling {

WebGpuProfiler::WebGpuProfiler(webgpu::WebGpuContext& webgpu_context)
    : webgpu_context_{webgpu_context} {
}

bool WebGpuProfiler::StartProfiling(TimePoint tp) {
  webgpu_context_.StartProfiling(tp);
  return true;
}

void WebGpuProfiler::EndProfiling(TimePoint tp, Events& events) {
  webgpu_context_.EndProfiling(tp, events);
}

}  // namespace profiling
}  // namespace onnxruntime
