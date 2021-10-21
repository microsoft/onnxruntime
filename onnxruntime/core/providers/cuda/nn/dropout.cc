// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/nn/dropout.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    12, 12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2),
    Dropout);

ONNX_OPERATOR_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", ALL_IEEE_FLOAT_TENSOR_TYPES)
        .TypeConstraint("T1", ALL_IEEE_FLOAT_TENSOR_TYPES)
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2),
    Dropout);

Status Dropout::ComputeInternal(OpKernelContext* context) const {
  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  const TensorShape& shape = X->Shape();
  const int64_t N = shape.Size();

  //Get Y_data
  auto Y = context->Output(0, shape);

  //Get mask_data
  auto mask = context->Output(1, shape);
  ORT_ENFORCE(!mask || mask->Shape().Size() == N);

  //Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(1);
  if (ratio) {
    utils::MLTypeCallDispatcher<ALL_IEEE_FLOAT_DATA_TYPES> t_disp(ratio->GetElementType());
    t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  const Tensor* training_mode = context->Input<Tensor>(2);
  //Check for inference mode.
  if ((0 == ratio_data) ||(training_mode == nullptr || *(training_mode->Data<bool>()) == false)) {
    const void* X_data = X->DataRaw();
    void* Y_data = Y->MutableDataRaw();
    if (Y_data != X_data) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y_data, X_data, X->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream()));
    }

    // If mask is requested, return all 1s.
    if (mask != nullptr) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(mask->MutableData<bool>(), true, N * sizeof(bool), Stream()));
    }

    return Status::OK();
  }

  IAllocatorUniquePtr<bool> temp_mask_buffer{};  // buffer to use if mask is not provided
  bool* const mask_data = [this, N, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableData<bool>();
    temp_mask_buffer = GetScratchBuffer<bool>(N);
    return temp_mask_buffer.get();
  }();

  PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();

  utils::MLTypeCallDispatcher<ALL_IEEE_FLOAT_DATA_TYPES> t_disp(X->GetElementType());
  t_disp.Invoke<DropoutComputeImpl>(GetDeviceProp(), Stream(), N, ratio_data, generator, *X, *Y, mask_data);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
