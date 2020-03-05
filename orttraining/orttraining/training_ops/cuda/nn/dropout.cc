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


static DropoutGenerator& GetGenerator() {
  // This generator is shared by all Dropouts.
  static DropoutGenerator generator(static_cast<uint64_t>(utils::GetStaticRandomSeed()));
  return generator;
}

template <typename T1, typename T2>
Status Dropout<T1, T2>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T1>::MappedType CudaT;
  typedef typename ToCudaType<T2>::MappedType CudaT2;

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
    ratio_data = static_cast<float>(*reinterpret_cast<const CudaT2*>(ratio->template Data<T2>()));
  } else {
    ratio_data = default_ratio_;
  }
  ORT_ENFORCE(ratio_data >= 0.0f && ratio_data < 1.0f);

  DropoutKernelImpl(GetDeviceProp(), N, ratio_data, generator_ != nullptr ? *generator_.get() : GetGenerator(), X_data, Y_data, mask_data);

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
    ratio_data = static_cast<float>(*reinterpret_cast<const T2*>(ratio->template Data<T2>()));
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
