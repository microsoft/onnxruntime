// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/vulkan/vulkan_utils.h"

namespace onnxruntime {
class VulkanExecutionProvider;

namespace vulkan {
class VulkanKernel : public OpKernel {
 public:
  explicit VulkanKernel(const OpKernelInfo& info)
      : OpKernel(info),
        exec_{*VULKAN_EXEC_PROVIDER_FROM_INFO(info)} {
  }

 private:
  const VulkanExecutionProvider& exec_;
};

}  // namespace vulkan
}  // namespace onnxruntime
