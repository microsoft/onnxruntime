// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)

#include <webgpu/webgpu_glfw.h>

#include "core/common/common.h"
#include "core/providers/webgpu/webgpu_pix_frame_generator.h"

namespace onnxruntime {
namespace webgpu {

WebGpuPIXFrameGenerator::WebGpuPIXFrameGenerator(wgpu::Instance instance, wgpu::Device device) {
  // Trivial window size for surface texture creation and provide frame concept for PIX.
  static constexpr uint32_t kWidth = 512u;
  static constexpr uint32_t kHeight = 512u;

  if (!glfwInit()) {
    ORT_ENFORCE("Failed to init glfw for PIX capture");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  window_ =
      glfwCreateWindow(kWidth, kHeight, "WebGPU window", nullptr, nullptr);

  ORT_ENFORCE(window_ != nullptr, "PIX Capture: Failed to create Window for capturing frames.");

  surface_ = wgpu::glfw::CreateSurfaceForWindow(instance, window_);
  ORT_ENFORCE(surface_.Get() != nullptr, "PIX Capture: Failed to create surface for capturing frames.");

  wgpu::TextureFormat format;
  wgpu::SurfaceCapabilities capabilities;
  surface_.GetCapabilities(device.GetAdapter(), &capabilities);
  format = capabilities.formats[0];

  wgpu::SurfaceConfiguration config;
  config.presentMode = capabilities.presentModes[0];
  config.device = device;
  config.format = format;
  config.width = kWidth;
  config.height = kHeight;

  surface_.Configure(&config);
}

void WebGpuPIXFrameGenerator::GeneratePIXFrame() {
  ORT_ENFORCE(surface_.Get() != nullptr, "PIX Capture: Cannot do present on null surface for capturing frames");
  wgpu::SurfaceTexture surfaceTexture;
  surface_.GetCurrentTexture(&surfaceTexture);

  // Call present to trigger dxgi_swapchain present. PIX
  // take this as a frame boundary.
  surface_.Present();
}

WebGpuPIXFrameGenerator::~WebGpuPIXFrameGenerator() {
  if (surface_.Get()) {
    surface_.Unconfigure();
  }

  if (window_) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
}

}  // namespace webgpu
}  // namespace onnxruntime
#endif  // ENABLE_PIX_FOR_WEBGPU_EP
