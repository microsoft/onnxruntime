// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "include/ncnn/layer.h"
#include "include/ncnn/command.h"

#include "core/common/gsl.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/kernel_def_builder.h"
#include "core/graph/constants.h"
#include "core/providers/vulkan/vulkan_execution_provider.h"

namespace onnxruntime {
namespace vulkan {
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

#define ONNX_OPERATOR_VULKAN_KERNEL_CLASS_NAME(op, version) \
  ONNX_OPERATOR_KERNEL_CLASS_NAME(kVulkanExecutionProvider, kOnnxDomain, version, op)

#define ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL_CLASS_NAME(op, since_version, end_version) \
  ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kVulkanExecutionProvider, kOnnxDomain, since_version, end_version, op)

#define REGISTER_ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL(op, since_version, end_version, builder, ...) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(op, kOnnxDomain, since_version, end_version,                     \
                                    kVulkanExecutionProvider, builder, __VA_ARGS__)

#define REGISTER_ONNX_OPERATOR_VULKAN_KERNEL(op, version, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(op, kOnnxDomain, version, kVulkanExecutionProvider, builder, __VA_ARGS__)

#define RETURN_IF_NCNN_ERROR(function)                               \
  do {                                                               \
    int ret = function;                                              \
    ORT_RETURN_IF(ret != 0, "Error calling ", #function, ": ", ret); \
  } while (0)

const VulkanExecutionProvider& GetVulkanExecutionProvider(const onnxruntime::OpKernelInfo& info);

ncnn::Mat TensorToMat(const Tensor& tensor);
ncnn::VkMat TensorToVkMat(const Tensor& tensor, ncnn::VkAllocator& allocator);

// apply packing logic that VkCompute::record_upload uses
ncnn::VkMat TensorToVkMatWithPacking(const Tensor& tensor, ncnn::VkAllocator& allocator,
                                     const ncnn::VulkanDevice& device, const ncnn::Option& options);

// get input/output shape hints
std::tuple<std::vector<ncnn::Mat>, std::vector<ncnn::Mat>> GetLayerShapeHints(const Node& node);

struct LayerPipeline {
  LayerPipeline(ncnn::Layer& layer, const ncnn::Option& options,
                const std::vector<ncnn::Mat>& input_shape_hints = {},
                const std::vector<ncnn::Mat>& output_shape_hints = {})
      : layer_(&layer),
        options_{&options} {
    layer_->bottom_shapes = input_shape_hints;
    layer_->top_shapes = output_shape_hints;

    ORT_ENFORCE(layer_->create_pipeline(*options_) == 0, "Failed to create pipeline");
    // TODO: There's no check on the actual call to `create_pipeline` being successful in the NCNN code.
    // e.g. sigmoid_vulkan.cpp has
    //         pipeline_sigmoid->create(LayerShaderType::sigmoid, opt, specializations);
    // We could override the create_pipeline in each layer (it's a virtual method) to plug in checks but it would need
    // to be on a per-layer basis as the variable name/s for the pipeline/s is specific to the layer.
    // A simple check would be that pipeline->shader_info is populated.
  }

  ~LayerPipeline() {
    auto result = layer_->destroy_pipeline(*options_);
    if (result != 0) {
      LOGS_DEFAULT(ERROR) << "Failed to destroy pipeline. Error code: " << result;
    }
  }

 private:
  ncnn::Layer* layer_{nullptr};
  const ncnn::Option* options_{nullptr};
};
}  // namespace vulkan
}  // namespace onnxruntime
