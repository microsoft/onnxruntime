// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/vulkan/vulkan_utils.h"

namespace onnxruntime {
class Node;
class VulkanExecutionProvider;

namespace logging {
class Logger;
}

namespace vulkan {
class KomputeTensor;
;
class VulkanKernel {
 public:
  virtual ~VulkanKernel() = default;

  // Do we have an implementation in Vulkan that supports this node?
  static bool IsSupported(const GraphViewer& graph_viewer, const Node& node, const logging::Logger& logger);

  struct ValueIndexes : std::unordered_map<std::string_view, int32_t> {
    int32_t Add(const NodeArg& def) {
      // use -1 for missing inputs/outputs.
      int32_t idx = def.Exists() ? gsl::narrow_cast<int32_t>(size()) : -1;
      (*this)[def.Name()] = idx;

      return idx;
    }
  };

  // Create and initialize the VulkanKernel for the Node.
  static Status Create(const VulkanExecutionProvider& vulkan_ep,
                       const GraphViewer* graph_viewer,
                       const onnxruntime::Node& node,
                       ValueIndexes& value_indexes,
                       std::unique_ptr<VulkanKernel>& kernel);

  // convenience method for static kernel usage
  static Status Create(const OpKernelInfo& info, std::unique_ptr<VulkanKernel>& kernel) {
    ValueIndexes value_indexes;

    return Create(GetVulkanExecutionProvider(info), nullptr, info.node(), value_indexes, kernel);
  }

  // static kernel usage
  virtual Status ComputeImpl(OpKernelContext& /*context*/) const {
    ORT_NOT_IMPLEMENTED("ComputeImpl is not implemented for ", node_.OpType());
  }

  using NodeArgToKpTensorMap = std::unordered_map<const NodeArg*, std::shared_ptr<kp::Tensor>>;

  // Add constant initializers that need to be uploaded to the device to initializers_to_upload by creating with
  // manager.tensor or manager.tensorT. This allows pre-packing to happen in the operation implementation.
  //
  // NOTE: this isn't necessarily all the constant inputs to the node - some might be provided to the shader as
  // specialization constants or push constants.
  virtual void ProcessConstantInitializers(
      const GraphViewer& /*graph_viewer*/, kp::Manager& /*manager*/,
      NodeArgToKpTensorMap& /*initializers_to_upload*/) const {
  }

  virtual Status CreateKernel(kp::Manager& /*manager*/, NodeArgToKpTensorMap& /*initializers*/) {
    ORT_NOT_IMPLEMENTED("Kernel must override");
  }

  virtual Status Execute(kp::Manager& /*manager*/, kp::Sequence& /*sequence*/,
                         NodeArgToKpTensorMap& /*values*/) const {
    ORT_NOT_IMPLEMENTED("Kernel must override");
  }

  // WARNING: This is invalid post-setup in a compiled model as the node is in the GraphViewer for the partition
  // and will be removed after IExecutionProvider::Compile completes.
  const onnxruntime::Node& Node() const { return node_; }

 protected:
  explicit VulkanKernel(const VulkanExecutionProvider& vulkan_ep, const onnxruntime::Node& node)
      : vulkan_ep_{vulkan_ep}, node_{node} {
  }

  const logging::Logger& Logger() const { return *vulkan_ep_.GetLogger(); }

 private:
  const VulkanExecutionProvider& vulkan_ep_;
  const onnxruntime::Node& node_;
};

}  // namespace vulkan
}  // namespace onnxruntime
