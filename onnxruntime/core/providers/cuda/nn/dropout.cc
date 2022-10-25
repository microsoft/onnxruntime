// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/nn/dropout.h"

#include "core/providers/cuda/nn/dropout_impl.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

namespace {

template <typename T>
struct GetRatioDataImpl {
  void operator()(const Tensor* ratio, float& ratio_data) const {
    ratio_data = static_cast<float>(*(ratio->Data<T>()));
    ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f, "ratio_data is outside range [0, 1)");
  }
};

template <typename T>
struct DropoutComputeImpl {
  void operator()(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N, const int64_t mask_element_count,
                  const float ratio_data, PhiloxGenerator& generator, const Tensor& X, Tensor& Y, void* mask_data,
                  bool use_bitmask) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    const CudaT* X_data = reinterpret_cast<const CudaT*>(X.Data<T>());
    CudaT* Y_data = reinterpret_cast<CudaT*>(Y.MutableData<T>());

    DropoutKernelImpl<CudaT>(prop, stream, N, mask_element_count, ratio_data, generator, X_data, Y_data, mask_data,
                             use_bitmask);
  }
};

}  // namespace

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Dropout, kOnnxDomain, 12, 12, kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T", DataTypeImpl::AllIEEEFloatTensorTypes())
                                      .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes())
                                      .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                                      .InputMemoryType(OrtMemTypeCPUInput, 1)
                                      .InputMemoryType(OrtMemTypeCPUInput, 2),
                                  Dropout<false>);

ONNX_OPERATOR_KERNEL_EX(Dropout, kOnnxDomain, 13, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>())
                            .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())
                            .InputMemoryType(OrtMemTypeCPUInput, 1)
                            .InputMemoryType(OrtMemTypeCPUInput, 2),
                        Dropout<false>);

template <bool UseBitmask>
Status Dropout<UseBitmask>::ComputeInternal(OpKernelContext* context) const {
  // Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  if (!X) return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  const TensorShape& shape = X->Shape();
  const int64_t N = shape.Size();

  // Get Y_data
  auto Y = context->Output(0, shape);

  // Get mask_data
  Tensor* mask = nullptr;
  int64_t mask_element_count = N;
  if (UseBitmask) {
    mask_element_count = (N + kNumBitsPerBitmaskElement - 1) / kNumBitsPerBitmaskElement;
    mask = context->Output(1, {mask_element_count});
  } else {
    mask = context->Output(1, shape);
  }

  ORT_ENFORCE(!mask || mask->Shape().Size() == mask_element_count);

  // Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(1);
  if (ratio) {
    utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(ratio->GetElementType());
    t_disp.Invoke<GetRatioDataImpl>(ratio, ratio_data);
  }

  const Tensor* training_mode = context->Input<Tensor>(2);
  // Check for inference mode.
  if (ratio_data == 0.f || !training_mode || !(*(training_mode->Data<bool>()))) {
    const void* X_data = X->DataRaw();
    void* Y_data = Y->MutableDataRaw();
    if (Y_data != X_data) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(Y_data, X_data, X->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream()));
    }

    // If mask is requested, return all 1s.
    if (mask) {
      if (UseBitmask) {
        CUDA_RETURN_IF_ERROR(
            cudaMemsetAsync(mask->MutableDataRaw(), -1, mask_element_count * sizeof(BitmaskElementType), Stream()));
      } else {
        CUDA_RETURN_IF_ERROR(
            cudaMemsetAsync(mask->MutableData<bool>(), true, mask_element_count * sizeof(bool), Stream()));
      }
    }

    return Status::OK();
  }

  IAllocatorUniquePtr<void> temp_mask_buffer{};  // buffer to use if mask is not provided
  void* const mask_data = [this, mask_element_count, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableDataRaw();
    temp_mask_buffer =
        GetScratchBuffer<void>(mask_element_count * (UseBitmask ? sizeof(BitmaskElementType) : sizeof(bool)));
    return temp_mask_buffer.get();
  }();

  PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();

  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(X->GetElementType());
  t_disp.Invoke<DropoutComputeImpl>(GetDeviceProp(), Stream(), N, mask_element_count, ratio_data, generator, *X, *Y,
                                    mask_data, UseBitmask);

  return Status::OK();
}

// Instantiation for Dropout.
template class Dropout<false>;
template class Dropout<true>;

}  // namespace cuda
}  // namespace onnxruntime
