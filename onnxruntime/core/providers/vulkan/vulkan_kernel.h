// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/vulkan/vulkan_utils.h"

namespace onnxruntime {
class Node;
class VulkanExecutionProvider;

namespace vulkan {
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
  static Status Create(const OpKernelInfo& info,
                       std::unique_ptr<VulkanKernel>& kernel) {
    ValueIndexes value_indexes;
    for (const auto def : info.node().InputDefs()) {
      value_indexes.Add(*def);
    }

    return Create(GetVulkanExecutionProvider(info), nullptr, info.node(), value_indexes, kernel);
  }

  // static kernel usage
  virtual Status ComputeImpl(OpKernelContext& /*context*/) const {
    ORT_NOT_IMPLEMENTED("ComputeImpl is not implemented for ", node_.OpType());
  }

  const onnxruntime::Node& Node() const { return node_; }
  const ncnn::Layer& Layer() const { return *ncnn_layer_; }

 protected:
  explicit VulkanKernel(const VulkanExecutionProvider& vulkan_ep, const onnxruntime::Node& node)
      : vulkan_ep_{vulkan_ep}, node_{node} {
  }

  // override if you need to map node.OpType() to a different NCNN layer name
  // see <build output dir>\_deps\ncnn-build\src\layer_registry.h for layer names
  virtual std::string_view GetNcnnLayerName() const { return node_.OpType(); }

  // default implementation that does not require parameters to be passed in.
  // override to setup ParamDict and call BaseInit
  //
  // TODO: Depending on whether we go with static kernels or compiling we need to provide a way to check if a value
  // is a constant initializer, and whether it is scalar or not (binary elementwise op optimization. maybe others).
  // If we're compiling we can use the GraphViewer for the fused node to get the TensorProto.
  // If we're using static kernels we can use TryGetConstantInput from OpKernelInfo but that returns a Tensor or
  // OrtValue.
  // Compiling seems lower friction so using GraphViewer for now and skipping optimizations in binary_elementwise.cc
  // if using static kernels.
  virtual Status CreateNcnnKernel(const GraphViewer* /*graph_viewer*/, ValueIndexes& value_indexes) {
    return SetupNcnnLayer(value_indexes);
  }

  // create ncnn_layer_, setup the layer shape hints, create the pipeline and populate value_indexes for the node.
  Status SetupNcnnLayer(ValueIndexes& value_indexes, const ncnn::ParamDict& params = {});

  const ncnn::Option& NcnnOptions() const { return vulkan_ep_.NcnnOptions(); }
  const ncnn::VulkanDevice& Device() const { return vulkan_ep_.Device(); }

 private:
  const VulkanExecutionProvider& vulkan_ep_;
  const onnxruntime::Node& node_;
  std::unique_ptr<ncnn::Layer> ncnn_layer_;
};

}  // namespace vulkan
}  // namespace onnxruntime
