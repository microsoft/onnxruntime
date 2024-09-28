// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace profiling {

WebGPUProfiler::WebGPUProfiler(int context_id)
    : webgpu_context_{webgpu::WebGpuContextFactory::GetContext(context_id)} {
}

bool WebGPUProfiler::StartProfiling(TimePoint) {
    webgpu_context_.StartProfiling("default");
    return true;
}

void WebGPUProfiler::EndProfiling(TimePoint, Events&) {
    webgpu_context_.EndProfiling();
}

}  // namespace profiling
}  // namespace onnxruntime
