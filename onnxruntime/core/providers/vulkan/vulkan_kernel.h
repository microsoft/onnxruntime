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
        vulkan_ep_{GetVulkanExecutionProvider(info)} {
  }

 protected:
  const ncnn::Option& NcnnOptions() const { return vulkan_ep_.NcnnOptions(); }
  const ncnn::VulkanDevice& Device() const { return vulkan_ep_.Device(); }

 private:
  const VulkanExecutionProvider& vulkan_ep_;
};

}  // namespace vulkan
}  // namespace onnxruntime
