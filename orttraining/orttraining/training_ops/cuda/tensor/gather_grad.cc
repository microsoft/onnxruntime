// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/gather_grad.h"
#include "orttraining/training_ops/cuda/tensor/gather_grad_impl.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    GatherGrad,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    GatherGrad);

namespace {
template <typename T, typename Tin>
Status CallGatherGradImpl(
    const CudaKernel& cuda_kernel,
    int64_t num_weights, int64_t stride, int64_t num_inputs, int64_t param_itrs,
    const Tensor& grad, const Tensor& indices,
    Tensor& output) {
  using CudaT = typename ToCudaType<T>::MappedType;

  const T* grad_data = grad.template Data<T>();
  T* output_data = output.template MutableData<T>();
  const Tin* indices_data = indices.template Data<Tin>();

  GatherGradImpl(
      cuda_kernel,
      reinterpret_cast<const CudaT*>(grad_data),
      indices_data,
      indices.Shape().Size(),
      num_weights,
      stride,
      reinterpret_cast<CudaT*>(output_data),
      num_inputs,
      param_itrs);

  return Status::OK();
}

template <typename T>
Status DispatchToGatherGradImplByTin(
    MLDataType tin_data_type,
    const CudaKernel& cuda_kernel,
    int64_t num_weights, int64_t stride, int64_t num_inputs, int64_t param_itrs,
    const Tensor& grad, const Tensor& indices,
    Tensor& output) {
  if (utils::IsPrimitiveDataType<int32_t>(tin_data_type)) {
    return CallGatherGradImpl<T, int32_t>(
        cuda_kernel, num_weights, stride, num_inputs, param_itrs, grad, indices, output);
  } else if (utils::IsPrimitiveDataType<int64_t>(tin_data_type)) {
    return CallGatherGradImpl<T, int64_t>(
        cuda_kernel, num_weights, stride, num_inputs, param_itrs, grad, indices, output);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GatherGrad unsupported Tin type: ", tin_data_type);
}

Status DispatchToGatherGradImpl(
    MLDataType t_data_type, MLDataType tin_data_type,
    const CudaKernel& cuda_kernel,
    int64_t num_weights, int64_t stride, int64_t num_inputs, int64_t param_itrs,
    const Tensor& grad, const Tensor& indices,
    Tensor& output) {
  if (utils::IsPrimitiveDataType<float>(t_data_type)) {
    return DispatchToGatherGradImplByTin<float>(
        tin_data_type, cuda_kernel, num_weights, stride, num_inputs, param_itrs, grad, indices, output);
  } else if (utils::IsPrimitiveDataType<MLFloat16>(t_data_type)) {
    return DispatchToGatherGradImplByTin<MLFloat16>(
        tin_data_type, cuda_kernel, num_weights, stride, num_inputs, param_itrs, grad, indices, output);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GatherGrad unsupported T type: ", t_data_type);
}
}  // namespace

Status GatherGrad::ComputeInternal(OpKernelContext* context) const {
  const Tensor* shape = context->Input<Tensor>(0);
  const TensorShape data_shape(shape->template Data<int64_t>(), shape->Shape().Size());
  const Tensor* indices = context->Input<Tensor>(1);
  const Tensor* grad = context->Input<Tensor>(2);

  Tensor* output = context->Output(0, data_shape);
  CUDA_RETURN_IF_ERROR(cudaMemset(output->MutableDataRaw(), 0, output->SizeInBytes()));
  MLDataType T_type = grad->DataType();
  MLDataType Tin_type = indices->DataType();

  const auto axis = HandleNegativeAxis(axis_, data_shape.NumDimensions());
  const int64_t stride = data_shape.SizeFromDimension(axis + 1);
  const int64_t num_weights = data_shape.Size() / stride;
  const int64_t num_inputs = data_shape.SizeFromDimension(axis);
  const int64_t param_itrs = data_shape.SizeFromDimension(0) / num_inputs;

  return DispatchToGatherGradImpl(
      T_type, Tin_type, *this,
      num_weights, stride, num_inputs, param_itrs,
      *grad, *indices, *output);
}

}  // namespace cuda
}  // namespace onnxruntime
