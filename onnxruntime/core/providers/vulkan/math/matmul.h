// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/vulkan/vulkan_kernel.h"

namespace onnxruntime {
class Node;
class VulkanExecutionProvider;

namespace vulkan {
class MatMulKernel : VulkanKernel {
 public:
  static bool IsSupported(const GraphViewer& graph_viewer, const onnxruntime::Node& node,
                          const logging::Logger& logger) {
    return true;
  }

  static std::unique_ptr<VulkanKernel> Create(const VulkanExecutionProvider& vulkan_ep,
                                              const GraphViewer& graph_viewer,
                                              const onnxruntime::Node& node) {
    return std::unique_ptr<VulkanKernel>(new MatMulKernel(vulkan_ep, graph_viewer, node));
  }

 private:
  MatMulKernel(const VulkanExecutionProvider& vulkan_ep,
               const GraphViewer& graph_viewer,
               const onnxruntime::Node& node);

  Status SetupParamDict(const GraphViewer& graph_viewer, ncnn::ParamDict& params) override;

  Status SetupConstantInitializers(const GraphViewer& graph_viewer, ncnn::Layer& layer) override;

  Status UploadConstantInitializers(ncnn::VkTransfer& cmd, ncnn::Option& upload_options) override;

  bool constant_A_{false};
  bool constant_B_{false};
  bool use_inner_product_{false};

  std::optional<std::vector<float>> transposed_b_;
};

}  // namespace vulkan
}  // namespace onnxruntime
