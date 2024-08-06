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
                       const GraphViewer& graph_viewer,
                       const onnxruntime::Node& node,
                       ValueIndexes& value_indexes,
                       std::unique_ptr<VulkanKernel>& kernel);

  virtual Status UploadConstantInitializers(ncnn::VkTransfer& cmd, ncnn::Option& upload_options) {
    // TODO: Do we need to support masked options?
    // int uret = layers[i]->upload_model(cmd, get_masked_option(opt_upload, layers[i]->featmask));

    RETURN_IF_NCNN_ERROR(ncnn_layer_->upload_model(cmd, upload_options));

    return Status::OK();
  }

  const onnxruntime::Node& Node() const { return node_; }

  ncnn::Layer& Layer() { return *ncnn_layer_; }
  const ncnn::Layer& Layer() const { return *ncnn_layer_; }

 protected:
  explicit VulkanKernel(const VulkanExecutionProvider& vulkan_ep, const onnxruntime::Node& node)
      : vulkan_ep_{vulkan_ep}, node_{node} {
  }

  // override if you need to map node.OpType() to a different NCNN layer name
  // see <build output dir>\_deps\ncnn-build\src\layer_registry.h for layer names
  virtual std::string_view GetNcnnLayerName() const { return node_.OpType(); }

  // default implementation that does not require parameters to be passed in to the NCNN layer.
  // override to setup ParamDict
  virtual Status SetupParamDict(const GraphViewer& /*graph_viewer*/, ncnn::ParamDict& /*params*/) {
    return Status::OK();
  }

  virtual Status SetupConstantInitializers(const GraphViewer& /*graph_viewer*/, ValueIndexes& /*value_indexes*/) {
    // Populate ncnn::Mat members of the specific NCNN Layer derived class with constant initializers if applicable.
    // Add any constant initializers to value_indexes.
    return Status::OK();
  }

  // override if there are CPU based ncnn::Mat members in the NCNN base layer that can be freed after the pipeline
  // is created
  virtual Status CreatePipeline() {
    RETURN_IF_NCNN_ERROR(ncnn_layer_->create_pipeline(vulkan_ep_.NcnnOptions()));
    return Status::OK();
  }

  // create ncnn_layer_, setup the layer shape hints, create the pipeline and populate value_indexes for the node.
  Status SetupNcnnLayer(const GraphViewer& graph_viewer, ValueIndexes& value_indexes);

  const ncnn::Option& NcnnOptions() const { return vulkan_ep_.NcnnOptions(); }
  const ncnn::VulkanDevice& Device() const { return vulkan_ep_.Device(); }
  const logging::Logger& Logger() const { return *vulkan_ep_.GetLogger(); }

 private:
  const VulkanExecutionProvider& vulkan_ep_;
  const onnxruntime::Node& node_;
  std::unique_ptr<ncnn::Layer> ncnn_layer_;
  ncnn::ParamDict params_;
};

}  // namespace vulkan
}  // namespace onnxruntime
