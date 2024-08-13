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
  static bool IsSupported(bool use_kompute, const GraphViewer& graph_viewer, const onnxruntime::Node& node,
                          const logging::Logger& logger);

  static std::unique_ptr<VulkanKernel> Create(const VulkanExecutionProvider& vulkan_ep,
                                              bool use_kompute,
                                              const GraphViewer& graph_viewer,
                                              const onnxruntime::Node& node) {
    return std::unique_ptr<VulkanKernel>(new MatMulKernel(vulkan_ep, use_kompute, graph_viewer, node));
  }

  // static kernel usage.
  Status ComputeImpl(OpKernelContext& context) const override;

  void SetPrepackedB(std::unique_ptr<Tensor> tensor) {
    transposed_b_tensor_ = std::move(tensor);
  }

 private:
  MatMulKernel(const VulkanExecutionProvider& vulkan_ep,
               bool use_kompute,
               const GraphViewer& graph_viewer,
               const onnxruntime::Node& node);

  std::string_view GetNcnnLayerName() const { return use_inner_product_ ? "InnerProduct" : "Gemm"; }

  Status SetupNcnnParamDict(const GraphViewer& graph_viewer, ncnn::ParamDict& params) override;

  Status SetupNcnnConstantInitializers(const GraphViewer& graph_viewer, ValueIndexes& value_indexes) override;

  Status UploadNcnnConstantInitializers(ncnn::VkTransfer& cmd, ncnn::Option& upload_options) override;

  Status CreateNcnnPipeline() override;

  void KomputeProcessConstantInitializers(
      const GraphViewer& graph_viewer, kp::Manager& manager,
      std::unordered_map<const NodeArg*, std::shared_ptr<kp::Tensor>>& initializers_to_upload) const override;

  Status KomputeExecute(kp::Manager& manager, kp::Sequence& sequence,
                        std::unordered_map<const NodeArg*, std::shared_ptr<kp::Tensor>>& values) const override;
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
  std::unique_ptr<Tensor> transposed_b_tensor_;
};

// wrapper to use as OpKernel while testing both paths
class MatMul : public OpKernel {
 public:
  explicit MatMul(const OpKernelInfo& info) : OpKernel(info) {
    ORT_THROW_IF_ERROR(VulkanKernel::Create(info, kernel_));
    matmul_kernel_ = reinterpret_cast<MatMulKernel*>(kernel_.get());
  }

  Status Compute(OpKernelContext* context) const override {
    return kernel_->ComputeImpl(*context);
  }

 private:
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed, /*out*/ PrePackedWeights* prepacked_weights) override;

  std::unique_ptr<VulkanKernel> kernel_;
  MatMulKernel* matmul_kernel_;
  std::unique_ptr<Tensor> transposed_b_;
};
}  // namespace vulkan
}  // namespace onnxruntime
