// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gather_elements_grad.h"
#include "core/providers/cuda/tensor/scatter_elements_impl.h"
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
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    GatherElementsGrad);

template <typename T>
struct GatherElementsGrad::ScatterAddComputeImpl {
  Status operator()(const Tensor* updates_tensor,
                    const Tensor* indices_tensor,
                    Tensor* output_tensor,
                    const int rank,
                    TArray<int64_t>& buffer_input_dims,
                    TArray<int64_t>& buffer_input_strides,
                    const int64_t indices_size,
                    TArray<int64_t>& buffer_indices_dims,
                    TArray<fast_divmod>& fdm_indices_strides,
                    const int axis) const {
    T* output_data = output_tensor->template MutableData<T>();
    const T* update_data = updates_tensor->template Data<T>();
    typedef typename ToCudaType<T>::MappedType CudaT;

    MLDataType Tin_type = indices_tensor->DataType();
    if (utils::IsPrimitiveDataType<int32_t>(Tin_type)) {
      const int32_t* indices_data = indices_tensor->template Data<int32_t>();
      return GatherElementsGradImpl(
          rank,
          buffer_input_dims,
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
          rank,
          buffer_input_dims,
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
  const Tensor* shape = context->Input<Tensor>(0);
  const TensorShape data_shape(shape->template Data<int64_t>(), shape->Shape().Size());

  const int axis = static_cast<int>(HandleNegativeAxis(axis_, data_shape.NumDimensions()));

  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto* updates_tensor = context->Input<Tensor>(2);

  const auto& indices_dims = indices_tensor->Shape().GetDims();
  const int64_t indices_size = indices_tensor->Shape().Size();
  const auto& updates_dims = updates_tensor->Shape().GetDims();
  if (indices_dims.size() != updates_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Indices and updates must have the same rank");
  }

  for (size_t i = 0; i < indices_dims.size(); ++i) {
    if (indices_dims[i] != updates_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices vs updates dimensions differs at position=", i,
                             " ", indices_dims[i], " vs ", updates_dims[i]);
    }
  }

  // According to the spec the rank of ind/upd shall be the same as input(data)
  // and we also want to make sure that the dimensions of the of the ind/upd do not
  // exceed that of the input
  const auto& input_dims = data_shape.GetDims();
  if (input_dims.size() != indices_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices must have the same rank as Input. Indices rank=",
                           indices_dims.size(), ". Input rank=", input_dims.size());
  }

  for (size_t i = 0; i < input_dims.size(); ++i) {
    if (input_dims[i] < indices_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices dim=", indices_dims[i], " at pos=", i,
                             " is greater than input dim=", input_dims[i]);
    }
  }

  int rank = (int)input_dims.size();
  Tensor* output_tensor = context->Output(0, data_shape);

  TArray<int64_t> buffer_input_dims(input_dims);
  TensorPitches input_strides(input_dims);
  TArray<int64_t> buffer_input_strides(input_strides);

  TArray<int64_t> buffer_indices_dims(indices_dims);
  TArray<fast_divmod> fdm_indices_strides(rank);
  TensorPitches indices_strides(indices_dims);
  for (auto i = 0; i < rank; i++) {
    fdm_indices_strides[i] = fast_divmod(static_cast<int>(indices_strides[i]));
  }

  utils::MLTypeCallDispatcherRet<Status, ScatterAddComputeImpl, float, MLFloat16>
      t_disp(updates_tensor->GetElementType());
  return t_disp.Invoke(updates_tensor, indices_tensor, output_tensor, rank,
                       buffer_input_dims, buffer_input_strides, indices_size,
                       buffer_indices_dims, fdm_indices_strides, axis);
}

}  // namespace cuda
}  // namespace onnxruntime
