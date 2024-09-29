// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace profiling {

WebGPUProfiler::WebGPUProfiler(int context_id)
    : webgpu_context_{webgpu::WebGpuContextFactory::GetContext(context_id)} {
}

bool WebGPUProfiler::StartProfiling(TimePoint ts) {
  webgpu_context_.StartProfiling(ts);
  return true;
}

void WebGPUProfiler::EndProfiling(TimePoint ts, Events& events) {
  webgpu_context_.EndProfiling(ts, events);
}

}  // namespace profiling
}  // namespace onnxruntime
