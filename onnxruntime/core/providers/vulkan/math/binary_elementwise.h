// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "include/ncnn/layer/binaryop.h"

#include "core/framework/op_kernel.h"
#include "core/providers/vulkan/vulkan_kernel.h"

namespace onnxruntime {
class Node;
class VulkanExecutionProvider;
namespace vulkan {

class BinaryElementwiseKernel : VulkanKernel {
 public:
  static bool IsSupported(bool use_kompute, const GraphViewer&, const onnxruntime::Node& node,
                          const logging::Logger& logger) {
    // VulkanKernel does the overall check for data types that are supported

    if (use_kompute) {
      LOGS(logger, VERBOSE) << "BinaryElementwiseKernel Kompute is not supported yet";
      return false;
    }

    static const std::unordered_set<std::string_view> supported_ops = {
        "Add",
        "Sub",
        "Mul",
        "Div",
    };

    if (supported_ops.count(node.OpType()) == 0) {
      LOGS(logger, VERBOSE) << "BinaryElementwiseKernel op_type: " << node.OpType() << " is not supported.";
      return false;
    }

    return true;
  }

// this setup is to avoid a big if/else if based on the node.OpType() string.
#define BEK_CREATE(name, ncnn_op_type)                                                        \
  static std::unique_ptr<VulkanKernel> Create##name(const VulkanExecutionProvider& vulkan_ep, \
                                                    bool use_kompute,                         \
                                                    const GraphViewer* graph_viewer,          \
                                                    const onnxruntime::Node& node) {          \
    return Create(ncnn_op_type, vulkan_ep, use_kompute, graph_viewer, node);                  \
  }

  BEK_CREATE(Add, ncnn::BinaryOp::Operation_ADD)
  BEK_CREATE(Sub, ncnn::BinaryOp::Operation_SUB)
  BEK_CREATE(Mul, ncnn::BinaryOp::Operation_MUL)
  BEK_CREATE(Div, ncnn::BinaryOp::Operation_DIV)
#undef BEK_CREATE

 private:
  static std::unique_ptr<VulkanKernel> Create(ncnn::BinaryOp::OperationType op_type,
                                              const VulkanExecutionProvider& vulkan_ep,
                                              bool use_kompute,
                                              const GraphViewer&,
                                              const onnxruntime::Node& node) {
    return std::unique_ptr<VulkanKernel>(new BinaryElementwiseKernel(op_type, vulkan_ep, use_kompute, node));
  }

  BinaryElementwiseKernel(ncnn::BinaryOp::OperationType ncnn_op_type,
                          const VulkanExecutionProvider& vulkan_ep,
                          bool use_kompute,
                          const onnxruntime::Node& node)
      : VulkanKernel{vulkan_ep, use_kompute, node},
        op_type_{ncnn_op_type} {
  }

  Status SetupNcnnParamDict(const GraphViewer& graph_viewer, ncnn::ParamDict& params) override;

  std::string_view GetNcnnLayerName() const override { return "BinaryOp"; }

  enum Params {
    kOperationType = 0,  // ncnn::BinaryOp::OperationType
    kWithScalar = 1,     // is the `b` input a scalar?
    kScalarValue = 2,    // float value for `b` input if kWithScalar is set to 1
  };

  const ncnn::BinaryOp::OperationType op_type_;
  bool has_scalar_input_{false};
};

}  // namespace vulkan
}  // namespace onnxruntime
