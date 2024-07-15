
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/activation/activations.h"

#include "ncnn-src/src/layer/vulkan/sigmoid_vulkan.h"

#include "core/providers/vulkan/vulkan_utils.h"

namespace onnxruntime {
namespace vulkan {

#define REGISTER_VERSIONED_KERNEL(op, since_version, end_version)                                    \
  REGISTER_ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL(                                                    \
      op, since_version, end_version,                                                                \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      op);

#define REGISTER_KERNEL(op, since_version)                                                           \
  REGISTER_ONNX_OPERATOR_VULKAN_KERNEL(                                                              \
      op, since_version,                                                                             \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      op);

REGISTER_VERSIONED_KERNEL(Sigmoid, 6, 12);
REGISTER_KERNEL(Sigmoid, 13);

Status SigmoidKernel::ComputeImpl(OpKernelContext& context) const {
  const Tensor& X = *context.Input<Tensor>(0);
  Tensor& Y = *context.Output(0, X.Shape());

  const auto& shape = X.Shape();
  const int64_t size = shape.Size();
  if (size == 0) {
    return Status::OK();
  }

  const auto& ncnn_options = NcnnOptions();
  const ncnn::VulkanDevice& device = Device();

  ncnn::VkCompute cmd(&device);  // TODO: This needs to be at a higher level so we can delay the submit_and_wait

  ncnn::VkMat src = TensorToVkMatWithPacking(X, *ncnn_options.blob_vkallocator, device, ncnn_options);
  ncnn::VkMat dst = TensorToVkMatWithPacking(Y, *ncnn_options.blob_vkallocator, device, ncnn_options);

  RETURN_IF_NCNN_ERROR(Layer().forward(src, dst, cmd, ncnn_options));

  // TODO: Investigate when/where we need barriers/waits.
  // c.f. with CUDA where we submit all the operations and only wait when we need to go back to CPU.
  // Do we need a shared VkCompute instance in the EP to do that? Does the data transfer also need to use that?
  RETURN_IF_NCNN_ERROR(cmd.submit_and_wait());

  return Status::OK();
}

bool ClipKernel::IsSupported(const GraphViewer& graph_viewer, const onnxruntime::Node& node,
                             const logging::Logger& logger) {
  if (node.SinceVersion() < 11) {
    // min/max are attributes
    return true;
  }

  const auto& input_defs = node.InputDefs();
  const auto num_inputs = input_defs.size();
  const NodeArg* min = num_inputs > 1 ? input_defs[1] : nullptr;
  const NodeArg* max = num_inputs > 2 ? input_defs[2] : nullptr;

  // min/max if provided must be constant. we can use the default values if they're missing.
  bool supported =
      (min == nullptr || min->Exists() == false || graph_viewer.GetConstantInitializer(min->Name()) != nullptr) &&
      (max == nullptr || max->Exists() == false || graph_viewer.GetConstantInitializer(max->Name()) != nullptr);

  if (!supported) {
    LOGS(logger, VERBOSE) << "Clip min/max must be constant";
  }

  return supported;
}
}  // namespace vulkan
}  // namespace onnxruntime
