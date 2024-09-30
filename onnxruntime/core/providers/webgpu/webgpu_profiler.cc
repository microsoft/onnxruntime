// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace profiling {

WebGPUProfiler::WebGPUProfiler(int context_id)
    : webgpu_context_{webgpu::WebGpuContextFactory::GetContext(context_id)} {
}

bool WebGPUProfiler::StartProfiling(TimePoint tp) {
  webgpu_context_.StartProfiling(tp);
  return true;
}

void WebGPUProfiler::EndProfiling(TimePoint tp, Events& events) {
  webgpu_context_.EndProfiling(tp, events);
}

}  // namespace profiling
}  // namespace onnxruntime
