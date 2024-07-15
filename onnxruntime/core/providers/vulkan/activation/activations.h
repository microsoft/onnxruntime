// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/vulkan/vulkan_kernel.h"

namespace onnxruntime {
class Node;
class VulkanExecutionProvider;
namespace vulkan {

class SigmoidKernel : VulkanKernel {
 public:
  static bool IsSupported(const GraphViewer&, const onnxruntime::Node&, const logging::Logger&) {
    return true;
  }

  static std::unique_ptr<VulkanKernel> Create(const VulkanExecutionProvider& vulkan_ep,
                                              const onnxruntime::Node& node) {
    return std::unique_ptr<VulkanKernel>(new SigmoidKernel(vulkan_ep, node));
  }

  // static kernel usage.
  Status ComputeImpl(OpKernelContext& context) const override;

 private:
  SigmoidKernel(const VulkanExecutionProvider& vulkan_ep, const onnxruntime::Node& node)
      : VulkanKernel{vulkan_ep, node} {
  }
};

class ClipKernel : VulkanKernel {
 public:
  static bool IsSupported(const GraphViewer& graph_viewer, const onnxruntime::Node& node,
                          const logging::Logger& logger);

  static std::unique_ptr<VulkanKernel> Create(const VulkanExecutionProvider& vulkan_ep,
                                              const onnxruntime::Node& node) {
    return std::unique_ptr<VulkanKernel>(new ClipKernel(vulkan_ep, node));
  }

  // std::string_view GetNcnnLayerName() const override {
  // if we add a Relu6 layer to NCNN we need this to plug in calling that for Clip(0, 6)
  // }

  // static kernel usage.
  // Status ComputeImpl(OpKernelContext& context) const override;

 private:
  ClipKernel(const VulkanExecutionProvider& vulkan_ep,
             const onnxruntime::Node& node)
      : VulkanKernel{vulkan_ep, node} {
  }
};

class Sigmoid : public OpKernel {
 public:
  explicit Sigmoid(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(VulkanKernel::Create(info, kernel_));
  }

  Status Compute(OpKernelContext* context) const override {
    return kernel_->ComputeImpl(*context);
  }

 private:
  std::unique_ptr<VulkanKernel> kernel_;
};

}  // namespace vulkan
}  // namespace onnxruntime
