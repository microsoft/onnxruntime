// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/vulkan/vulkan_kernel.h"

namespace kp {
class Algorithm;
}

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
                                              const GraphViewer* graph_viewer,
                                              const onnxruntime::Node& node) {
    assert(graph_viewer);  // expecting compile only usage
    return std::unique_ptr<VulkanKernel>(new MatMulKernel(vulkan_ep, *graph_viewer, node));
  }

  // static kernel usage.
  Status ComputeImpl(OpKernelContext& context) const override;

  void SetPrepackedB(std::unique_ptr<Tensor> tensor) {
    transposed_b_tensor_ = std::move(tensor);
  }

 private:
  MatMulKernel(const VulkanExecutionProvider& vulkan_ep,
               const GraphViewer& graph_viewer,
               const onnxruntime::Node& node);

  void ProcessConstantInitializers(const GraphViewer& graph_viewer, kp::Manager& manager,
                                   NodeArgToKpTensorMap& initializers_to_upload) const override;

  Status CreateKernel(kp::Manager& manager, NodeArgToKpTensorMap& initializers) override;
  Status Execute(kp::Manager& manager, kp::Sequence& sequence, NodeArgToKpTensorMap& values) const override;

  struct InputInfo {
    InputInfo(const GraphViewer& graph_viewer, const onnxruntime::Node& node, const logging::Logger& logger);

    bool constant_A;
    bool constant_B;
    bool have_shape_A;
    bool have_shape_B;
    std::vector<int64_t> shape_A;
    std::vector<int64_t> shape_B;
    const NodeArg* arg_A;
    const NodeArg* arg_B;
  };

  const InputInfo input_info_;

  std::unique_ptr<Tensor> transposed_b_tensor_;

  mutable std::shared_ptr<kp::Algorithm> kompute_kernel_;
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
