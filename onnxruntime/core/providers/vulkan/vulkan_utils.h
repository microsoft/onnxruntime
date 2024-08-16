// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"

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

}  // namespace vulkan
}  // namespace onnxruntime
