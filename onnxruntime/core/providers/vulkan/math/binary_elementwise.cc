
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/vulkan/math/binary_elementwise.h"

#include "ncnn-src/src/layer/binaryop.h"

#include "core/optimizer/initializer.h"

namespace onnxruntime {
namespace vulkan {

#define REGISTER_VERSIONED_KERNEL(op, since_version, end_version)                                    \
  REGISTER_ONNX_VERSIONED_OPERATOR_VULKAN_KERNEL(                                                    \
      op, since_version, end_version,                                                                \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      BinaryElementwise);

#define REGISTER_KERNEL(op, since_version)                                                           \
  REGISTER_ONNX_OPERATOR_VULKAN_KERNEL(                                                              \
      op, since_version,                                                                             \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      BinaryElementwise);

REGISTER_VERSIONED_KERNEL(Mul, 7, 12);
REGISTER_VERSIONED_KERNEL(Mul, 13, 13);
REGISTER_KERNEL(Mul, 14);

Status BinaryElementwiseKernel::CreateNcnnKernel(const GraphViewer* graph_viewer, ValueIndexes& value_indexes) {
  ncnn::ParamDict params;
  params.set(Params::kOperationType, op_type_);

  const auto& node = Node();
  const auto& input_defs = node.InputDefs();

  // Use the NCNN scalar input optimization if `b` is a scalar.
  // TODO: If we wanted to support `a` being a scalar it requires a lot more work as we'd be flipping the inputs around
  // and other things like adding shape hints for inputs would need to be aware of this. Probably easier to change
  // the model instead of adding complexity here for what may be a rare use case.
  const auto& b_arg = *input_defs[1];
  const auto* b = graph_viewer ? graph_viewer->GetConstantInitializer(b_arg.Name(), true) : nullptr;

  const auto is_scalar_input = [](const NodeArg& input) {
    const auto* shape = input.Shape();
    return (shape && shape->dim_size() == 1 && shape->dim(0).has_dim_value() && shape->dim(0).dim_value() == 1);
  };

  if (b && is_scalar_input(b_arg)) {
    params.set(Params::kWithScalar, 1);
    const auto& initializer = *graph_viewer->GetConstantInitializer(b_arg.Name());
    Initializer data(initializer);
    // we only support float currently. need to handle other data types
    assert(data.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    params.set(Params::kScalarValue, *data.data<float>());
  }

  return VulkanKernel::SetupNcnnLayer(value_indexes, params);
}

Status BinaryElementwiseKernel::ComputeImpl(OpKernelContext& context) const {
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

}  // namespace vulkan
}  // namespace onnxruntime
