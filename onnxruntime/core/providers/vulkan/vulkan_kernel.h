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
  static bool IsSupported(bool use_kompute, const GraphViewer& graph_viewer, const Node& node,
                          const logging::Logger& logger);

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
                       bool use_kompute,
                       const GraphViewer* graph_viewer,
                       const onnxruntime::Node& node,
                       ValueIndexes& value_indexes,
                       std::unique_ptr<VulkanKernel>& kernel);

  // convenience method for static kernel usage
  static Status Create(const OpKernelInfo& info, std::unique_ptr<VulkanKernel>& kernel) {
    ValueIndexes value_indexes;

    return Create(GetVulkanExecutionProvider(info), true, nullptr, info.node(), value_indexes, kernel);
  }

  // static kernel usage
  virtual Status ComputeImpl(OpKernelContext& /*context*/) const {
    ORT_NOT_IMPLEMENTED("ComputeImpl is not implemented for ", node_.OpType());
  }

  virtual Status UploadNcnnConstantInitializers(ncnn::VkTransfer& cmd, ncnn::Option& upload_options) {
    // TODO: Do we need to support masked options?
    // int uret = layers[i]->upload_model(cmd, get_masked_option(opt_upload, layers[i]->featmask));

    RETURN_IF_NCNN_ERROR(ncnn_layer_->upload_model(cmd, upload_options));

    return Status::OK();
  }

  // Add constant initializers that need to be uploaded to the device to initializers_to_upload by creating with
  // manager.tensor or manager.tensorT. This allows pre-packing to happen in the operation implementation.
  //
  // NOTE: this isn't necessarily all the constant inputs to the node - some might be provided to the shader as
  // specialization constants or push constants.
  virtual void KomputeProcessConstantInitializers(
      const GraphViewer& /*graph_viewer*/, kp::Manager& /*manager*/,
      std::unordered_map<const NodeArg*, std::shared_ptr<kp::Tensor>>& /*initializers_to_upload*/) const {
  }

  virtual Status KomputeExecute(kp::Manager& /*manager*/, kp::Sequence& /*sequence*/,
                                std::unordered_map<const NodeArg*, std::shared_ptr<kp::Tensor>>& /*values*/) const {
    ORT_NOT_IMPLEMENTED("Kernel must override");
  }

  const onnxruntime::Node& Node() const { return node_; }

  ncnn::Layer& NcnnLayer() { return *ncnn_layer_; }
  const ncnn::Layer& NcnnLayer() const { return *ncnn_layer_; }

 protected:
  explicit VulkanKernel(const VulkanExecutionProvider& vulkan_ep, bool use_kompute,
                        const onnxruntime::Node& node)
      : vulkan_ep_{vulkan_ep}, use_kompute_{use_kompute}, node_{node} {
  }

  // override if you need to map node.OpType() to a different NCNN layer name
  // see <build output dir>\_deps\ncnn-build\src\layer_registry.h for layer names
  virtual std::string_view GetNcnnLayerName() const { return node_.OpType(); }

  // default implementation that does not require parameters to be passed in to the NCNN layer.
  // override to setup ParamDict
  virtual Status SetupNcnnParamDict(const GraphViewer& /*graph_viewer*/, ncnn::ParamDict& /*params*/) {
    return Status::OK();
  }

  virtual Status SetupNcnnConstantInitializers(const GraphViewer& /*graph_viewer*/, ValueIndexes& /*value_indexes*/) {
    // Populate ncnn::Mat members of the specific NCNN Layer derived class with constant initializers if applicable.
    // Add any constant initializers that are NOT directly provided to the kernel to value_indexes as they need to be
    // passed in as an input.
    return Status::OK();
  }

  // override if there are CPU based ncnn::Mat members in the NCNN base layer that can be freed after the pipeline
  // is created
  virtual Status CreateNcnnPipeline() {
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
  const bool use_kompute_;
  const onnxruntime::Node& node_;

  // NCNN
  std::unique_ptr<ncnn::Layer> ncnn_layer_;
  ncnn::ParamDict ncnn_params_;

  // Kompute
};

}  // namespace vulkan
}  // namespace onnxruntime
