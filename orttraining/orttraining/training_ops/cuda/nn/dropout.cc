// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/random_seed.h"
#include "orttraining/training_ops/cuda/nn/dropout.h"

#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T1, T2)                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      Dropout,                                                       \
      kOnnxDomain,                                                   \
      12,                                                            \
      T1##_##T2,                                                     \
      kCudaExecutionProvider,                                        \
      KernelDefBuilder()                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())    \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()) \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                    \
          .InputMemoryType<OrtMemTypeCPUInput>(2),                   \
      Dropout<T1, T2, false>);

REGISTER_KERNEL_TYPED(MLFloat16, MLFloat16)
REGISTER_KERNEL_TYPED(MLFloat16, float)
REGISTER_KERNEL_TYPED(MLFloat16, double)
REGISTER_KERNEL_TYPED(float, MLFloat16)
REGISTER_KERNEL_TYPED(float, float)
REGISTER_KERNEL_TYPED(float, double)
REGISTER_KERNEL_TYPED(double, MLFloat16)
REGISTER_KERNEL_TYPED(double, float)
REGISTER_KERNEL_TYPED(double, double)

#define REGISTER_TRAINABLE_KERNEL_TYPED(T1, T2)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      TrainableDropout,                                              \
      kOnnxDomain,                                                   \
      9,                                                             \
      T1##_##T2,                                                     \
      kCudaExecutionProvider,                                        \
      KernelDefBuilder()                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())    \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())   \
          .InputMemoryType<OrtMemTypeCPUInput>(1),                   \
      Dropout<T1, T2, true>);

// Temporary for backward compatibility, will eventually get rid of TrainableDropout when PyTorch exporter will move to 
// opset-12.
REGISTER_TRAINABLE_KERNEL_TYPED(MLFloat16, MLFloat16)
REGISTER_TRAINABLE_KERNEL_TYPED(MLFloat16, float)
REGISTER_TRAINABLE_KERNEL_TYPED(MLFloat16, double)
REGISTER_TRAINABLE_KERNEL_TYPED(float, MLFloat16)
REGISTER_TRAINABLE_KERNEL_TYPED(float, float)
REGISTER_TRAINABLE_KERNEL_TYPED(float, double)
REGISTER_TRAINABLE_KERNEL_TYPED(double, MLFloat16)
REGISTER_TRAINABLE_KERNEL_TYPED(double, float)
REGISTER_TRAINABLE_KERNEL_TYPED(double, double)

template <typename T1, typename T2, bool trainable_dropout>
Status Dropout<T1, T2, trainable_dropout>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T1>::MappedType CudaT;

  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  const TensorShape& shape = X->Shape();
  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T1>());
  const int64_t N = shape.Size();

  //Get Y_data
  auto Y = context->Output(0, shape);
  auto Y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T1>());

  //Get mask_data
  auto mask = context->Output(1, shape);
  ORT_ENFORCE(!mask || mask->Shape().Size() == N);

  const Tensor* training_mode = context->Input<Tensor>(2);
  //Check for inference mode.
  if (!trainable_dropout && (training_mode == nullptr || *(training_mode->Data<bool>()) == false)) {
    if (Y_data != X_data) {
      CUDA_CALL_THROW(cudaMemcpyAsync(Y_data, X_data, N * sizeof(T1), cudaMemcpyDeviceToDevice));
    }

    // If mask is requested, return all 1s.
    if (mask != nullptr) {
      ORT_ENFORCE(cudaMemset(mask->MutableData<bool>(), true, N * sizeof(bool)) == cudaSuccess);
    }

    return Status::OK();
  }

  IAllocatorUniquePtr<bool> temp_mask_buffer{};  // buffer to use if mask is not provided
  bool* const mask_data = [this, N, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableData<bool>();
    temp_mask_buffer = GetScratchBuffer<bool>(N);
    return temp_mask_buffer.get();
  }();

  //Get the ratio_data
  float ratio_data;
  auto ratio = context->Input<Tensor>(1);

  static_assert(std::is_same<T2, MLFloat16>::value || std::is_same<T2, float>::value || std::is_same<T2, double>::value,
                "T2 must be float16 or float or double");

  if (ratio) {
    ratio_data = static_cast<float>(*(ratio->template Data<T2>()));
  } else {
    ratio_data = default_ratio_;
  }
  ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f);

  PhiloxGenerator& generator = generator_ != nullptr ? *generator_.get() : PhiloxGenerator::Default();
  DropoutKernelImpl(GetDeviceProp(), N, ratio_data, generator, X_data, Y_data, mask_data);

  return Status::OK();
}

#define REGISTER_GRADIENT_KERNEL_TYPED(OpName, T1, T2)               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      OpName,                                                        \
      kMSDomain,                                                     \
      1,                                                             \
      T1##_##T2,                                                     \
      kCudaExecutionProvider,                                        \
      KernelDefBuilder()                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())    \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()) \
          .InputMemoryType<OrtMemTypeCPUInput>(2),                   \
      DropoutGrad<T1, T2>);

REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, MLFloat16, MLFloat16)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, MLFloat16, float)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, MLFloat16, double)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, float, MLFloat16)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, float, float)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, float, double)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, double, MLFloat16)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, double, float)
REGISTER_GRADIENT_KERNEL_TYPED(DropoutGrad, double, double)

// Temporary for backward compatibility, will eventually get rid of TrainableDropout when PyTorch exporter will move to
// opset-12.
REGISTER_GRADIENT_KERNEL_TYPED(TrainableDropoutGrad, MLFloat16, MLFloat16)
REGISTER_GRADIENT_KERNEL_TYPED(TrainableDropoutGrad, MLFloat16, float)
REGISTER_GRADIENT_KERNEL_TYPED(TrainableDropoutGrad, MLFloat16, double)
REGISTER_GRADIENT_KERNEL_TYPED(TrainableDropoutGrad, float, MLFloat16)
REGISTER_GRADIENT_KERNEL_TYPED(TrainableDropoutGrad, float, float)
REGISTER_GRADIENT_KERNEL_TYPED(TrainableDropoutGrad, float, double)
REGISTER_GRADIENT_KERNEL_TYPED(TrainableDropoutGrad, double, MLFloat16)
REGISTER_GRADIENT_KERNEL_TYPED(TrainableDropoutGrad, double, float)
REGISTER_GRADIENT_KERNEL_TYPED(TrainableDropoutGrad, double, double)

template <typename T1, typename T2>
Status DropoutGrad<T1, T2>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T1>::MappedType CudaT;

  auto dY = context->Input<Tensor>(0);
  const TensorShape& shape = dY->Shape();
  auto dY_data = reinterpret_cast<const CudaT*>(dY->template Data<T1>());
  const int64_t N = shape.Size();

  auto mask = context->Input<Tensor>(1);
  ORT_ENFORCE(mask->Shape().Size() == N);

  auto dX = context->Output(0, shape);
  auto dX_data = reinterpret_cast<CudaT*>(dX->template MutableData<T1>());
  float ratio_data;
  auto ratio = context->Input<Tensor>(2);

  static_assert(std::is_same<T2, MLFloat16>::value || std::is_same<T2, float>::value || std::is_same<T2, double>::value,
                "T2 must be float16 or float or double");

  if (ratio) {
    ratio_data = static_cast<float>(*(ratio->template Data<T2>()));
  } else {
    ratio_data = default_ratio_;
  }
  ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f);

  const bool* mask_data = mask->template Data<bool>();
  DropoutGradientKernelImpl(N, dY_data, mask_data, ratio_data, dX_data);

  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime
