// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/gather_elements_grad.h"
#include "orttraining/training_ops/cuda/tensor/gather_elements_grad_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    GatherElementsGrad,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(1)  // 'GatherElements' data shape needs to be on CPU
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                        DataTypeImpl::GetTensorType<int64_t>()}),
    GatherElementsGrad);

template <typename T>
struct GatherElementsGrad::ComputeImpl {
  Status operator()(cudaStream_t stream,
                    const Tensor* dY,
                    const Tensor* indices_tensor,
                    Tensor* dX,
                    const int rank,
                    TArray<int64_t>& buffer_output_dims,
                    TArray<int64_t>& buffer_input_strides,
                    const int64_t indices_size,
                    TArray<int64_t>& buffer_indices_dims,
                    TArray<fast_divmod>& fdm_indices_strides,
                    const int axis) const {
    T* output_data = dX->template MutableData<T>();
    const T* update_data = dY->template Data<T>();
    typedef typename ToCudaType<T>::MappedType CudaT;

    MLDataType Tin_type = indices_tensor->DataType();
    if (utils::IsPrimitiveDataType<int32_t>(Tin_type)) {
      const int32_t* indices_data = indices_tensor->template Data<int32_t>();
      return GatherElementsGradImpl(
          stream,
          rank,
          buffer_output_dims,
          buffer_input_strides,
          indices_data,
          indices_size,
          buffer_indices_dims,
          fdm_indices_strides,
          reinterpret_cast<const CudaT*>(update_data),
          axis,
          reinterpret_cast<CudaT*>(output_data));
    } else if (utils::IsPrimitiveDataType<int64_t>(Tin_type)) {
      const int64_t* indices_data = indices_tensor->template Data<int64_t>();
      return GatherElementsGradImpl(
          stream,
          rank,
          buffer_output_dims,
          buffer_input_strides,
          indices_data,
          indices_size,
          buffer_indices_dims,
          fdm_indices_strides,
          reinterpret_cast<const CudaT*>(update_data),
          axis,
          reinterpret_cast<CudaT*>(output_data));
    }

    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Type for Tin is not supported yet in ScatterElements.");
  }
};

Status GatherElementsGrad::ComputeInternal(OpKernelContext* context) const {
  const auto* dY = context->Input<Tensor>(0);
  const Tensor* shape = context->Input<Tensor>(1);
  const TensorShape data_shape(shape->template Data<int64_t>(), shape->Shape().Size());

  const int axis = static_cast<int>(HandleNegativeAxis(axis_, data_shape.NumDimensions()));

  const auto* indices_tensor = context->Input<Tensor>(2);

  const auto& indices_dims = indices_tensor->Shape().GetDims();
  const int64_t indices_size = indices_tensor->Shape().Size();
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
    if (output_dims[i] < indices_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices dim=", indices_dims[i], " at pos=", i,
                             " is greater than Output dim=", output_dims[i]);
    }
  }

  int rank = static_cast<int>(output_dims.size());
  Tensor* dX = context->Output(0, data_shape);
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(dX->MutableDataRaw(), 0, dX->SizeInBytes(), Stream()));

  TArray<int64_t> buffer_output_dims(output_dims);
  TensorPitches input_strides(output_dims);
  TArray<int64_t> buffer_input_strides(input_strides);

  TArray<int64_t> buffer_indices_dims(indices_dims);
  TArray<fast_divmod> fdm_indices_strides(rank);
  TensorPitches indices_strides(indices_dims);
  for (auto i = 0; i < rank; i++) {
    fdm_indices_strides[i] = fast_divmod(static_cast<int>(indices_strides[i]));
  }

  utils::MLTypeCallDispatcher<MLFloat16, float, double> t_disp(dY->GetElementType());
  return t_disp.InvokeRet<Status, ComputeImpl>(
      Stream(), dY, indices_tensor, dX, rank,
      buffer_output_dims, buffer_input_strides, indices_size,
      buffer_indices_dims, fdm_indices_strides, axis);
}

}  // namespace cuda
}  // namespace onnxruntime
