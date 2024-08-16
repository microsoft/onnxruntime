// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/vulkan_utils.h"

namespace onnxruntime {
namespace vulkan {

const VulkanExecutionProvider& GetVulkanExecutionProvider(const onnxruntime::OpKernelInfo& info) {
  return *static_cast<const VulkanExecutionProvider*>(info.GetExecutionProvider());
}

}  // namespace vulkan
}  // namespace onnxruntime
