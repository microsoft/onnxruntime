// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/gsl.h"
#include "core/providers/vulkan/vulkan_execution_provider.h"

#include "ncnn-src/src/layer.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/kernel_def_builder.h"
#include "core/graph/constants.h"

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

#define RETURN_IF_NCNN_ERROR(function, ...)                              \
  do {                                                                   \
    auto ret = function(__VA_ARGS__);                                    \
    ORT_RETURN_IF_NOT(ret == 0, "Error calling ", #function, ": ", ret); \
  } while (0)

const VulkanExecutionProvider& GetVulkanExecutionProvider(const onnxruntime::OpKernelInfo& info);

// Get the index of the layer in the ncnn model. Throws if not found.
int GetNcnnLayerIndex(const std::string& layer_name);

ncnn::Mat TensorToMat(const Tensor& tensor);
ncnn::VkMat TensorToVkMat(const Tensor& tensor, ncnn::VkAllocator& allocator);

struct LayerPipeline {
  LayerPipeline(ncnn::Layer& layer, const ncnn::Option& options) : layer_(&layer), options_{&options} {
    ORT_ENFORCE(layer_->create_pipeline(*options_) == 0, "Failed to create pipeline");
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
