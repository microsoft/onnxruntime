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
  static bool IsSupported(const onnxruntime::Node& /*node*/, const logging::Logger& /*logger*/) {
    // implement check here
    // - data type/s - VulkanKernel does the overall check for types that are supported
    // - any param values that are required to create the pipeline are constant initializers
    //   - we _could_ create a pipeline on a per-Run basis to handle this but we don't support that currently

    // If the return will be false, log the reason why

    return true;
  }

  static std::unique_ptr<VulkanKernel> Create(const VulkanExecutionProvider& vulkan_ep,
                                              const onnxruntime::Node& node,
                                              std::unique_ptr<ncnn::Layer> layer) {
    return std::unique_ptr<VulkanKernel>(new SigmoidKernel(vulkan_ep, node, std::move(layer)));
  }

  // static kernel usage.
  Status ComputeImpl(OpKernelContext& context) const override;

 private:
  SigmoidKernel(const VulkanExecutionProvider& vulkan_ep,
                const onnxruntime::Node& node,
                std::unique_ptr<ncnn::Layer> layer)
      : VulkanKernel{vulkan_ep, node, std::move(layer)} {
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

// class HardSigmoid final : public Sigmoid {
//  public:
//   explicit HardSigmoid(const OpKernelInfo& info) : Sigmoid(info) {}
//   ~HardSigmoid() = default;
//
//   Status Compute(OpKernelContext* context) const override { return Sigmoid::Compute(context); }
//
//   static bool IsSupported(const onnxruntime::Node& /*node*/, const logging::Logger& /*logger*/) {
//     // implement any non-data type checks here.
//     // log why nodes are not supported if rejecting
//     return true;
//   }
// };
}  // namespace vulkan
}  // namespace onnxruntime
