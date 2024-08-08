// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ncnn-src/src/layer/binaryop.h"

#include "core/framework/op_kernel.h"
#include "core/providers/vulkan/vulkan_kernel.h"

namespace onnxruntime {
class Node;
class VulkanExecutionProvider;
namespace vulkan {

class BinaryElementwiseKernel : VulkanKernel {
 public:
  static bool IsSupported(const GraphViewer&, const onnxruntime::Node&, const logging::Logger&) {
    // implement check here
    // - data type/s - VulkanKernel does the overall check for types that are supported
    // - any param values that are required to create the pipeline are constant initializers
    //   - we _could_ create a pipeline on a per-Run basis to handle this but we don't support that currently

    // If the return will be false, log the reason why
    return true;
  }

  // static kernel usage.
  Status ComputeImpl(OpKernelContext& context) const override;

#define BEK_CREATE(name, ncnn_op_type)                                                        \
  static std::unique_ptr<VulkanKernel> Create##name(const VulkanExecutionProvider& vulkan_ep, \
                                                    const onnxruntime::Node& node) {          \
    return Create(ncnn_op_type, vulkan_ep, node);                                             \
  }

  BEK_CREATE(Add, ncnn::BinaryOp::Operation_ADD)
  BEK_CREATE(Sub, ncnn::BinaryOp::Operation_SUB)
  BEK_CREATE(Mul, ncnn::BinaryOp::Operation_MUL)
  BEK_CREATE(Div, ncnn::BinaryOp::Operation_DIV)
#undef BEK_CREATE

 private:
  static std::unique_ptr<VulkanKernel> Create(ncnn::BinaryOp::OperationType op_type,
                                              const VulkanExecutionProvider& vulkan_ep,
                                              const onnxruntime::Node& node) {
    return std::unique_ptr<VulkanKernel>(new BinaryElementwiseKernel(op_type, vulkan_ep, node));
  }

  BinaryElementwiseKernel(ncnn::BinaryOp::OperationType ncnn_op_type,
                          const VulkanExecutionProvider& vulkan_ep,
                          const onnxruntime::Node& node)
      : VulkanKernel{vulkan_ep, node},
        op_type_{ncnn_op_type} {
  }

  Status CreateNcnnKernel(const GraphViewer* graph_viewer, ValueIndexes& value_indexes) override;

  std::string_view GetNcnnLayerName() const override { return "BinaryOp"; }

  enum Params {
    kOperationType = 0,  // ncnn::BinaryOp::OperationType
    kWithScalar = 1,     // is the `b` input a scalar?
    kScalarValue = 2,    // float value for `b` input if kWithScalar is set to 1
  };

  const ncnn::BinaryOp::OperationType op_type_;
  bool has_scalar_input_{false};
};

class BinaryElementwise : public OpKernel {
 public:
  explicit BinaryElementwise(const OpKernelInfo& info) : OpKernel(info) {
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
