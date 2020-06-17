// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/random_seed.h"
#include "orttraining/training_ops/cuda/nn/dropout.h"
#include "core/providers/cuda/nn/dropout.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
struct GetRatioDataImpl {
  void operator()(const Tensor* ratio, float& ratio_data) const {
    if (ratio) {
      ratio_data = static_cast<float>(*(ratio->template Data<T>()));
    }
    ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f, "ratio_data is outside range [0, 1)");
  }
};

#define REGISTER_TRAINABLE_KERNEL_TYPED(T1, T2)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      TrainableDropout,                                            \
      kOnnxDomain,                                                 \
      9,                                                           \
      T1##_##T2,                                                   \
      kCudaExecutionProvider,                                      \
      KernelDefBuilder()                                           \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())  \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>()) \
          .InputMemoryType<OrtMemTypeCPUInput>(1),                 \
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
  typedef typename ToCudaType<T1>::MappedType CudaT1;

  auto dY = context->Input<Tensor>(0);
  const TensorShape& shape = dY->Shape();
  auto dY_data = reinterpret_cast<const CudaT1*>(dY->template Data<T1>());
  const int64_t N = shape.Size();

  auto mask = context->Input<Tensor>(1);
  ORT_ENFORCE(mask->Shape().Size() == N);

  auto dX = context->Output(0, shape);
  auto dX_data = reinterpret_cast<CudaT1*>(dX->template MutableData<T1>());
  
  // float ratio_data;
  // auto ratio = context->Input<Tensor>(2);

  // static_assert(std::is_same<T2, MLFloat16>::value || std::is_same<T2, float>::value || std::is_same<T2, double>::value,
  //               "T2 must be float16 or float or double");

  // if (ratio) {
  //   ratio_data = static_cast<float>(*(ratio->template Data<T2>()));
  // } else {
  //   ratio_data = default_ratio_;
  // }
  // ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f);

  //Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(2);
  utils::MLTypeCallDispatcher<GetRatioDataImpl, float, MLFloat16, double> t_disp(ratio->GetElementType());
  t_disp.Invoke(ratio, ratio_data);

  const bool* mask_data = mask->template Data<bool>();
  DropoutGradientKernelImpl(N, dY_data, mask_data, ratio_data, dX_data);

  return Status::OK();
}

#define REGISTER_BIAS_DROPOUT_KERNEL_TYPED(T)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      BiasDropout,                                                       \
      kMSDomain,                                                         \
      1,                                                                 \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())     \
          .InputMemoryType<OrtMemTypeCPUInput>(3)                        \
          .InputMemoryType<OrtMemTypeCPUInput>(4),                       \
      BiasDropout<T>);

REGISTER_BIAS_DROPOUT_KERNEL_TYPED(MLFloat16)
REGISTER_BIAS_DROPOUT_KERNEL_TYPED(float)

template <typename T1>
Status BiasDropout<T1>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T1>::MappedType CudaT1;

  //Get X_data
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF_NOT(X, "X Input is not available.");

  const TensorShape& x_shape = X->Shape();
  auto X_data = reinterpret_cast<const CudaT1*>(X->template Data<T1>());
  const int64_t N = x_shape.Size();

  //Get bias_data
  const Tensor* bias = context->Input<Tensor>(1);
  if (bias == nullptr) return Status(common::ONNXRUNTIME, common::FAIL, "Bias input of BiasDropout is not available.");
  const TensorShape& bias_shape = bias->Shape();
  if (bias_shape.NumDimensions() != 1) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Bias input is not a 1D tensor.");
  }
  const int64_t dim = bias_shape[0];
  if (dim != x_shape.GetDims().back()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "Bias' dimension doesn't match input's last dimension.");
  }
  auto bias_data = reinterpret_cast<const CudaT1*>(bias->template Data<T1>());

  //Get residual_data
  const Tensor* residual = context->Input<Tensor>(2);
  const CudaT1* residual_data = nullptr;
  if (residual != nullptr) {
    const TensorShape& residual_shape = residual->Shape();
    if (residual_shape != x_shape) {
      return Status(common::ONNXRUNTIME, common::FAIL, "Residual input shape does not match X input shape.");
    }
    residual_data = reinterpret_cast<const CudaT1*>(residual->template Data<T1>());
  }

  //Get Y_data
  auto Y = context->Output(0, x_shape);
  auto Y_data = reinterpret_cast<CudaT1*>(Y->template MutableData<T1>());

  //Get mask_data
  auto mask = context->Output(1, x_shape);

  //Get the ratio_data
  float ratio_data = default_ratio_;
  auto ratio = context->Input<Tensor>(3);
  utils::MLTypeCallDispatcher<GetRatioDataImpl, float, MLFloat16, double> t_disp(ratio->GetElementType());
  t_disp.Invoke(ratio, ratio_data);

  IAllocatorUniquePtr<bool> temp_mask_buffer{};  // buffer to use if mask is not provided
  bool* const mask_data = [this, N, mask, &temp_mask_buffer]() {
    if (mask) return mask->MutableData<bool>();
    temp_mask_buffer = GetScratchBuffer<bool>(N);
    return temp_mask_buffer.get();
  }();

  const fast_divmod fdm_dim(gsl::narrow_cast<int>(dim));
  PhiloxGenerator& generator = generator_ ? *generator_ : PhiloxGenerator::Default();
  BiasDropoutKernelImpl(GetDeviceProp(), N, fdm_dim, ratio_data, generator, X_data, bias_data, residual_data, Y_data, mask_data);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
