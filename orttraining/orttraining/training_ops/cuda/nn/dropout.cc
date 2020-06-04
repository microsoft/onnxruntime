// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/random_seed.h"
#include "orttraining/training_ops/cuda/nn/dropout.h"

#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(OpName, Domain, VER, T1, T2, MemIndex)  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      OpName,                                                         \
      Domain,                                                         \
      VER,                                                            \
      T1##_##T2,                                                      \
      kCudaExecutionProvider,                                         \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())    \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())  \
          .InputMemoryType<OrtMemTypeCPUInput>(MemIndex),             \
      OpName<T1, T2>);

REGISTER_KERNEL_TYPED(Dropout, kOnnxDomain, 12, MLFloat16, MLFloat16, 1)
REGISTER_KERNEL_TYPED(Dropout, kOnnxDomain, 12, MLFloat16, float, 1)
REGISTER_KERNEL_TYPED(Dropout, kOnnxDomain, 12, MLFloat16, double, 1)
REGISTER_KERNEL_TYPED(Dropout, kOnnxDomain, 12, float, MLFloat16, 1)
REGISTER_KERNEL_TYPED(Dropout, kOnnxDomain, 12, float, float, 1)
REGISTER_KERNEL_TYPED(Dropout, kOnnxDomain, 12, float, double, 1)
REGISTER_KERNEL_TYPED(Dropout, kOnnxDomain, 12, double, MLFloat16, 1)
REGISTER_KERNEL_TYPED(Dropout, kOnnxDomain, 12, double, float, 1)
REGISTER_KERNEL_TYPED(Dropout, kOnnxDomain, 12, double, double, 1)


template <typename T1, typename T2>
Status Dropout<T1, T2>::ComputeInternal(OpKernelContext* context) const {
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

REGISTER_KERNEL_TYPED(DropoutGrad, kMSDomain, 1, MLFloat16, MLFloat16, 2)
REGISTER_KERNEL_TYPED(DropoutGrad, kMSDomain, 1, MLFloat16, float, 2)
REGISTER_KERNEL_TYPED(DropoutGrad, kMSDomain, 1, MLFloat16, double, 2)
REGISTER_KERNEL_TYPED(DropoutGrad, kMSDomain, 1, float, MLFloat16, 2)
REGISTER_KERNEL_TYPED(DropoutGrad, kMSDomain, 1, float, float, 2)
REGISTER_KERNEL_TYPED(DropoutGrad, kMSDomain, 1, float, double, 2)
REGISTER_KERNEL_TYPED(DropoutGrad, kMSDomain, 1, double, MLFloat16, 2)
REGISTER_KERNEL_TYPED(DropoutGrad, kMSDomain, 1, double, float, 2)
REGISTER_KERNEL_TYPED(DropoutGrad, kMSDomain, 1, double, double, 2)

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

#define REGISTER_BIAS_DROPOUT_KERNEL_TYPED(T1, T2)                   \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      BiasDropout,                                                   \
      kMSDomain,                                                     \
      1,                                                             \
      T1##_##T2,                                                     \
      kCudaExecutionProvider,                                        \
      KernelDefBuilder()                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())    \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()) \
          .InputMemoryType<OrtMemTypeCPUInput>(3)                    \
          .InputMemoryType<OrtMemTypeCPUInput>(4),                   \
      BiasDropout<T1, T2>);

REGISTER_BIAS_DROPOUT_KERNEL_TYPED(MLFloat16, MLFloat16)
REGISTER_BIAS_DROPOUT_KERNEL_TYPED(MLFloat16, float)
REGISTER_BIAS_DROPOUT_KERNEL_TYPED(float, MLFloat16)
REGISTER_BIAS_DROPOUT_KERNEL_TYPED(float, float)

template <typename T1, typename T2>
Status BiasDropout<T1, T2>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T1>::MappedType CudaT;
 
  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  if (X == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "X Input is not available.");
  const TensorShape& x_shape = X->Shape();
  auto X_data = reinterpret_cast<const CudaT*>(X->template Data<T1>());
  const int64_t N = x_shape.Size();

 //Get bias_data
  const Tensor* bias = context->Input<Tensor>(1);
  if (bias == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "Bias input of BiasDropout is not available.");
  const TensorShape& bias_shape = bias->Shape();
  if (bias_shape.NumDimensions() != 1) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Bias input is not a 1D tensor.");
  }
  const int64_t dim = bias_shape[0];
  if (dim != x_shape.GetDims().back()){
    return Status(common::ONNXRUNTIME, common::FAIL, "Bias' dimension doesn't match input's last dimension.");
  }
  auto bias_data = reinterpret_cast<const CudaT*>(bias->template Data<T1>());

  //Get residual_data
  const Tensor* residual = context->Input<Tensor>(2);
  const CudaT* residual_data = nullptr;
  if (residual != nullptr) {
    const TensorShape& residual_shape = residual->Shape();
    if (residual_shape != x_shape) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Residual input shape does not match X input shape.");
    }
    residual_data = reinterpret_cast<const CudaT*>(residual->template Data<T1>());
  }

  //Get Y_data
  auto Y = context->Output(0, x_shape);
  auto Y_data = reinterpret_cast<CudaT*>(Y->template MutableData<T1>());

  //Get mask_data
  auto mask = context->Output(1, x_shape);
  ORT_ENFORCE(!mask || mask->Shape().Size() == N);

  //Get the ratio_data
  float ratio_data;
  auto ratio = context->Input<Tensor>(3);

  static_assert(std::is_same<T2, MLFloat16>::value || std::is_same<T2, float>::value || std::is_same<T2, double>::value,
                "T2 must be float16 or float or double");

  if (ratio) {
    ratio_data = static_cast<float>(*(ratio->template Data<T2>()));
  } else {
    ratio_data = default_ratio_;
  }
  ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f);

  // const Tensor* training_mode = context->Input<Tensor>(4);
  // //Check for inference mode.
  // if (training_mode == nullptr || *(training_mode->Data<bool>()) == false) {
  //   if (Y_data != X_data) {
  //     CUDA_CALL_THROW(cudaMemcpyAsync(Y_data, X_data, N * sizeof(T1), cudaMemcpyDeviceToDevice));
  //   }

  //   // If mask is requested, return all 1s.
  //   if (mask != nullptr) {
  //     ORT_ENFORCE(cudaMemset(mask->MutableData<bool>(), true, N * sizeof(bool)) == cudaSuccess);
  //   }

  //   return Status::OK();
  // }

  IAllocatorUniquePtr<bool> temp_mask_buffer{};  // buffer to use if mask is not provided
  bool* const mask_data = [this, N, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableData<bool>();
    temp_mask_buffer = GetScratchBuffer<bool>(N);
    return temp_mask_buffer.get();
  }();

  const fast_divmod fdm_dim(gsl::narrow_cast<int>(dim));
  PhiloxGenerator& generator = generator_ != nullptr ? *generator_.get() : PhiloxGenerator::Default();
  BiasDropoutKernelImpl(GetDeviceProp(), N, fdm_dim, ratio_data, generator, X_data, bias_data, residual_data, Y_data, mask_data);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
