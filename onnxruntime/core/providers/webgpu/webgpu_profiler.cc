// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "webgpu_execution_provider.h"

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

void WebGpuProfiler::SetCaptureTool(onnxruntime::profiling::CaptureTool tool) {
  onnxruntime::profiling::EpProfiler::SetCaptureTool(tool);
  context_.SetCaptureTool(GetCaptureTool());
}

void WebGpuProfiler::StartCapture() {
  context_.StartCapture();
}

void WebGpuProfiler::EndCapture() {
  context_.EndCapture();
}

const std::unordered_set<onnxruntime::profiling::CaptureTool>& WebGpuProfiler::GetSupportedCaptureToolSet() {
  return context_.GetSupportedCaptureToolSet();
}

}  // namespace webgpu
}  // namespace onnxruntime
