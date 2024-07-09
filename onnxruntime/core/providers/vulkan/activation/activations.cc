
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/activation/activations.h"

#include "ncnn-src/src/layer/vulkan/sigmoid_vulkan.h"

#include "core/framework/tensorprotoutils.h"
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

Sigmoid::Sigmoid(const OpKernelInfo& info)
    : VulkanKernel(info),
      data_type_{info.GetInputType(0)->tensor_type().elem_type()},
      ncnn_index_{GetNcnnLayerIndex("Sigmoid")},
      ncnn_layer_{ncnn::create_layer_vulkan(ncnn_index_)} {
  ORT_ENFORCE(ncnn_layer_, "Failed to create NCNN layer.");
  ncnn_layer_->vkdev = &Device();

  // If the params don't rely on input data shapes we can set them here.
  // If for some reason an ONNX input values maps to a param we'd need to do this in SetupLayer and call from Compute
  // ncnn::ParamDict params;
  // ncnn_layer_->load_param(params);

  auto* tensorproto_shape = Node().InputDefs()[0]->Shape();
  if (tensorproto_shape) {
    TensorShape shape = utils::GetTensorShapeFromTensorShapeProto(*tensorproto_shape);
    if (shape.Size() > 0) {
      fixed_size_pipeline_.emplace(*ncnn_layer_, NcnnOptions());
    }
  }
}

void Sigmoid::SetupLayer() const {
  // load_model to set dims. nothing required for an activation though.
  // ncnn::ModelBin mb;            // set this up for layers that need weights
  // ncnn_layer_->load_model(mb);  // <-- this needs fixed sizes to be called

  // create_pipeline
  ncnn_layer_->create_pipeline(NcnnOptions());
}

Status Sigmoid::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);
  Tensor& Y = *context->Output(0, X.Shape());

  const auto& shape = X.Shape();
  const int64_t size = shape.Size();
  if (size == 0) {
    return Status::OK();
  }

  std::optional<LayerPipeline> layer_pipeline;
  if (!fixed_size_pipeline_) {
    // TODO: We can optimize looking up an existing pipeline in the cache here
    // Currently we go through Layer::create_pipeline which sets up a lot of things based on the input shape
    // followed by the pipeline cache lookup which hashes the GLSL, the x/y/z values and the specializations.
    // We should be able to do a simple match on the shape in this case.
    // This also needs thought about how we'll structure things to ship with bytecode. The matching might be
    // input shape + data type. We may also want to make the cache key configurable on a per operator basis
    // so we can make it as simple as possible whilst also supported more complex operators.
    layer_pipeline = LayerPipeline(*ncnn_layer_, NcnnOptions());
  }

  const auto& ncnn_options = NcnnOptions();
  ncnn::VkCompute cmd(&Device());

  if (X.DataRaw() == Y.DataRaw()) {
  } else {
  }
  ncnn::VkMat src = TensorToVkMat(X, *ncnn_options.blob_vkallocator);
  ncnn::VkMat dst = TensorToVkMat(Y, *ncnn_options.blob_vkallocator);

  RETURN_IF_NCNN_ERROR(ncnn_layer_->forward(src, dst, cmd, ncnn_options));

  // TODO: Investigate when/where we need barriers/waits.
  // c.f. with CUDA where we submit all the operations and only wait when we need to go back to CPU.
  // Do we need a shared VkCompute instance in the EP to do that? Does the data transfer also need to use that?
  RETURN_IF_NCNN_ERROR(cmd.submit_and_wait());

  return Status::OK();
}

Sigmoid::~Sigmoid() {
}

}  // namespace vulkan
}  // namespace onnxruntime
