// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/tensor/gather_elements_grad.h"
#include "orttraining/training_ops/cpu/tensor/gather_elements_grad_impl.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    GatherElementsGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    GatherElementsGrad);

#define TYPED_GRAD_FUNCTION_CALL(T)                                             \
  if (T_type == DataTypeImpl::GetType<T>()) {                                   \
    if (Tind_type == DataTypeImpl::GetType<int32_t>()) {                        \
      return GatherElementsGradImpl<int32_t, T>(indices_tensor, dY, axis, dX);  \
    }                                                                           \
    if (Tind_type == DataTypeImpl::GetType<int64_t>()) {                        \
      return GatherElementsGradImpl<int64_t, T>(indices_tensor, dY, axis, dX);  \
    }                                                                           \
  }

Status GatherElementsGrad::Compute(OpKernelContext* context) const {
  const auto* dY = context->Input<Tensor>(0);
  const Tensor* shape = context->Input<Tensor>(1);
  const TensorShape data_shape(shape->template Data<int64_t>(), shape->Shape().Size());

  const int axis = static_cast<int>(HandleNegativeAxis(axis_, data_shape.NumDimensions()));

  const auto* indices_tensor = context->Input<Tensor>(2);

  const auto& indices_dims = indices_tensor->Shape().GetDims();
  const auto& dY_dims = dY->Shape().GetDims();
  if (indices_dims.size() != dY_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Indices and dY must have the same rank");
  }

  for (size_t i = 0; i < indices_dims.size(); ++i) {
    if (indices_dims[i] != dY_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices vs dY dimensions differs at position=", i,
                             " ", indices_dims[i], " vs ", dY_dims[i]);
    }
  }

  // According to the spec the rank of ind/upd shall be the same as output(data)
  // and we also want to make sure that the dimensions of the of the ind/upd do not
  // exceed that of the output
  const auto& output_dims = data_shape.GetDims();
  if (output_dims.size() != indices_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices must have the same rank as Output. Indices rank=",
                           indices_dims.size(), ". Output rank=", output_dims.size());
  }

  for (size_t i = 0; i < output_dims.size(); ++i) {
    // For all axes except the axis of interest, make sure that the corresponding 'indices' shape
    // value is within bounds of the corresponding 'data' shape.
    if (static_cast<int64_t>(i) != axis_ && output_dims[i] < indices_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices dim=", indices_dims[i], " at pos=", i,
                             " is greater than Output dim=", output_dims[i]);
    }
  }

  Tensor* dX = context->Output(0, data_shape);
  ORT_ENFORCE(dX);
  memset(dX->MutableDataRaw(), 0, dX->SizeInBytes());

  MLDataType T_type = dY->DataType();
  MLDataType Tind_type = indices_tensor->DataType();
  TYPED_GRAD_FUNCTION_CALL(float);
  TYPED_GRAD_FUNCTION_CALL(double);

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for T or Tind not supported yet in GatherElementsGrad.");
}

}  // namespace contrib
}  // namespace onnxruntime
