// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <utility>

#include "orttraining/training_ops/cuda/tensor/resize_grad.h"
#include "orttraining/training_ops/cuda/tensor/resize_grad_impl.h"

namespace onnxruntime::cuda {

#define REGISTER_RESIZEGRAD_KERNEL_TYPED(T)                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      ResizeGrad,                                                          \
      kMSDomain,                                                           \
      1,                                                                   \
      T,                                                                   \
      kCudaExecutionProvider,                                              \
      (*KernelDefBuilder::Create())                                        \
          .InputMemoryType(OrtMemTypeCPUInput, 2) /* Keep roi on CPU */    \
          .InputMemoryType(OrtMemTypeCPUInput, 3) /* Keep scales on CPU */ \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),          \
      ResizeGrad<T>);

REGISTER_RESIZEGRAD_KERNEL_TYPED(MLFloat16)
REGISTER_RESIZEGRAD_KERNEL_TYPED(float)
REGISTER_RESIZEGRAD_KERNEL_TYPED(double)

template <typename T>
Status ResizeGrad<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X = context->Input<Tensor>(1);
  const Tensor* scales = context->Input<Tensor>(3);

  ORT_ENFORCE(X->Shape().NumDimensions() == 4, "Expected input tensor to have 4 dimensions. Actual: ",
              X->Shape().NumDimensions());

  const auto get_scales_from_input = [](const Tensor* scales) {
    if (nullptr == scales) {
      return std::make_pair(std::optional<float>{}, std::optional<float>{});
    }

    ORT_ENFORCE(scales->Shape().Size() == 4, "There must be a scale for each dimension.");

    const auto* scales_data = scales->Data<float>();
    return std::make_pair(std::optional<float>{scales_data[2]}, std::optional<float>{scales_data[3]});
  };

  std::pair<std::optional<float>, std::optional<float>> scale_factors = get_scales_from_input(scales);

  Tensor* dX = context->Output(0, X->Shape());

  const int64_t batch_size = X->Shape()[0];
  const int64_t num_channels = X->Shape()[1];
  const int64_t output_height = dY->Shape()[2];
  const int64_t output_width = dY->Shape()[3];
  const int64_t input_height = X->Shape()[2];
  const int64_t input_width = X->Shape()[3];

  if (dX->Shape() == dY->Shape()) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dX->MutableDataRaw(), dY->DataRaw(), dY->SizeInBytes(), cudaMemcpyDeviceToDevice));
    return Status::OK();
  }

  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(dX->MutableDataRaw(), 0, dX->SizeInBytes(), Stream(context)));

  const bool align_corners = coordinate_transform_mode_ == ResizeCoordinateTransformationMode::ALIGN_CORNERS;
  const CudaT* dy_data = reinterpret_cast<const CudaT*>(dY->Data<T>());
  CudaT* dx_data = reinterpret_cast<CudaT*>(dX->MutableData<T>());

  ResizeGradImpl(Stream(context), input_height, input_width, output_height,
                 output_width, batch_size, num_channels, align_corners,
                 scale_factors.first, scale_factors.second,
                 dy_data, dx_data);

  return Status::OK();
}

}  // namespace onnxruntime::cuda
