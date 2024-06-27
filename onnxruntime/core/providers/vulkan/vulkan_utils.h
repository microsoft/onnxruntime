// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/vulkan/vulkan_execution_provider.h"

#define ONNX_VULKAN_OPERATOR_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kOnnxDomain, ver, kVulkanExecutionProvider, builder, __VA_ARGS__)

#define VULKAN_EXEC_PROVIDER_FROM_INFO(info) \
  static_cast<const VulkanExecutionProvider*>((info).GetExecutionProvider())

namespace onnxruntime {
namespace vulkan {
}  // namespace vulkan
}  // namespace onnxruntime
