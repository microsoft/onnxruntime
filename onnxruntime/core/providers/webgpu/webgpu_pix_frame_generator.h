// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
#include <GLFW/glfw3.h>
#endif  // ENABLE_PIX_FOR_WEBGPU_EP

#include <memory>

#include "core/providers/webgpu/webgpu_external_header.h"

namespace onnxruntime {

namespace webgpu {

// PIX(https://devblogs.microsoft.com/pix/introduction/) is a profiling tool
// provides by Microsoft. It has ability to do GPU capture to profile gpu
// behavior among different GPU vendors. It works on Windows only.
//
// GPU capture(present-to-present) provided by PIX uses present as a frame boundary to
// capture and generate a valid frame infos. But ORT WebGPU EP doesn't have any present logic
// and hangs PIX GPU Capture forever.
//
// To make PIX works with ORT WebGPU EP on Windows, WebGpuPIXFrameGenerator class includes codes
// to create a trivial window through glfw, config surface with Dawn device and call present in
// proper place to trigger frame boundary for PIX GPU Capture.
//
// WebGpuPIXFrameGenerator is an friend class because:
// - It should only be used in WebGpuContext class implementation.
// - It requires instance and device from WebGpuContext.
//
// The lifecycle of WebGpuPIXFrameGenerator instance should be nested into WebGpuContext lifecycle.
// WebGpuPIXFrameGenerator instance should be created during WebGpuContext creation and be destroyed during
// WebGpuContext destruction.
class WebGpuPIXFrameGenerator {
 public:
  WebGpuPIXFrameGenerator(wgpu::Instance instance, wgpu::Device device);
  ~WebGpuPIXFrameGenerator();
  void GeneratePIXFrame();

 private:
  void CreateSurface();
  wgpu::Surface surface_;
  GLFWwindow* window_;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuPIXFrameGenerator);
};

}  // namespace webgpu
}  // namespace onnxruntime
#endif  // ENABLE_PIX_FOR_WEBGPU_EP
