// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/vulkan/vulkan_kernel.h"

namespace ncnn {
class Mat;
}

namespace onnxruntime {
class Node;
class VulkanExecutionProvider;

namespace vulkan {
class MatMulKernel : VulkanKernel {
 public:
  static bool IsSupported(const GraphViewer& graph_viewer, const onnxruntime::Node& node,
                          const logging::Logger& logger);

  static std::unique_ptr<VulkanKernel> Create(const VulkanExecutionProvider& vulkan_ep,
                                              const GraphViewer& graph_viewer,
                                              const onnxruntime::Node& node) {
    return std::unique_ptr<VulkanKernel>(new MatMulKernel(vulkan_ep, graph_viewer, node));
  }

 private:
  MatMulKernel(const VulkanExecutionProvider& vulkan_ep,
               const GraphViewer& graph_viewer,
               const onnxruntime::Node& node);

  std::string_view GetNcnnLayerName() const { return use_inner_product_ ? "InnerProduct" : "Gemm"; }

  Status SetupParamDict(const GraphViewer& graph_viewer, ncnn::ParamDict& params) override;

  Status SetupConstantInitializers(const GraphViewer& graph_viewer, ValueIndexes& value_indexes) override;

  Status UploadConstantInitializers(ncnn::VkTransfer& cmd, ncnn::Option& upload_options) override;

  Status CreatePipeline() override;

  struct InputInfo {
    InputInfo(const GraphViewer& graph_viewer, const onnxruntime::Node& node, const logging::Logger& logger);

    bool constant_A;
    bool constant_B;
    bool have_shape_A;
    bool have_shape_B;
    std::vector<int64_t> shape_A;
    std::vector<int64_t> shape_B;
  };

  const InputInfo input_info_;
  const bool use_inner_product_;

  std::optional<ncnn::Mat> transposed_b_;
};

}  // namespace vulkan
}  // namespace onnxruntime
