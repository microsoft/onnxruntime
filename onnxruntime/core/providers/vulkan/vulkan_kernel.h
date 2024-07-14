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
  static bool IsSupported(const Node& node, const logging::Logger& logger);

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
                       const onnxruntime::Node& node,
                       ValueIndexes& value_indexes,
                       std::unique_ptr<VulkanKernel>& kernel);

  // convenience method for static kernel usage
  static Status Create(const OpKernelInfo& info, std::unique_ptr<VulkanKernel>& kernel) {
    ValueIndexes value_indexes;
    for (const auto def : info.node().InputDefs()) {
      value_indexes.Add(*def);
    }

    return Create(GetVulkanExecutionProvider(info), info.node(), value_indexes, kernel);
  }

  // static kernel usage
  virtual Status ComputeImpl(OpKernelContext& context) const = 0;

  const onnxruntime::Node& Node() const { return node_; }
  const ncnn::Layer& Layer() const { return *ncnn_layer_; }

 protected:
  explicit VulkanKernel(const VulkanExecutionProvider& vulkan_ep,
                        const onnxruntime::Node& node,
                        std::unique_ptr<ncnn::Layer> layer)
      : vulkan_ep_{vulkan_ep},
        node_{node},
        ncnn_layer_{std::move(layer)} {
  }

  // initialize the NCNN layer based on the Node
  virtual Status Init(ValueIndexes& value_indexes);

  const ncnn::Option& NcnnOptions() const { return vulkan_ep_.NcnnOptions(); }
  const ncnn::VulkanDevice& Device() const { return vulkan_ep_.Device(); }

 private:
  const VulkanExecutionProvider& vulkan_ep_;
  const onnxruntime::Node& node_;
  int32_t ncnn_index_{-1};
  std::unique_ptr<ncnn::Layer> ncnn_layer_;
};

}  // namespace vulkan
}  // namespace onnxruntime
