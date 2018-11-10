// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lrn.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      LRN,                                                                      \
      kOnnxDomain,                                                              \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LRN<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
LRN<T>::LRN(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t size;
  ONNXRUNTIME_ENFORCE(info.GetAttr<int64_t>("size", &size).IsOK());
  ONNXRUNTIME_ENFORCE(size > 0);
  ONNXRUNTIME_ENFORCE(size % 2 == 1);

  float alpha;
  float beta;
  ONNXRUNTIME_ENFORCE(info.GetAttr<float>("alpha", &alpha).IsOK());
  ONNXRUNTIME_ENFORCE(alpha > 0.0f);
  ONNXRUNTIME_ENFORCE(info.GetAttr<float>("beta", &beta).IsOK());
  ONNXRUNTIME_ENFORCE(beta > 0.0f);
  float bias = info.GetAttrOrDefault<float>("bias", 1.0f);

  ONNXRUNTIME_ENFORCE(norm_desc_.Set(
                            gsl::narrow_cast<uint32_t>(size),
                            static_cast<double>(alpha),
                            static_cast<double>(beta),
                            static_cast<double>(bias))
                  .IsOK());
}

template <typename T>
Status LRN<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* X = context->Input<Tensor>(0);

  auto rank = X->Shape().NumDimensions();
  if (rank != 4 && rank != 5)
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "cudnn LRN only supports 4D or 5D input");

  Tensor* Y = context->Output(0, X->Shape());

  CudnnTensor x_tensor;
  ONNXRUNTIME_RETURN_IF_ERROR(x_tensor.Set(X->Shape().GetDims(), CudnnTensor::GetDataType<CudaT>()));

  const auto one = Consts<CudaT>::One;
  const auto zero = Consts<CudaT>::Zero;

  CUDNN_RETURN_IF_ERROR(cudnnLRNCrossChannelForward(
      CudnnHandle(),
      norm_desc_,
      CUDNN_LRN_CROSS_CHANNEL_DIM1,
      &one,
      x_tensor,
      reinterpret_cast<const CudaT*>(X->template Data<T>()),
      &zero,
      x_tensor,
      reinterpret_cast<CudaT*>(Y->template MutableData<T>())));

  return Status::OK();
}

CudnnLRNDescriptor::CudnnLRNDescriptor() : desc_(nullptr) {
}

CudnnLRNDescriptor::~CudnnLRNDescriptor() {
  if (desc_) {
    cudnnDestroyLRNDescriptor(desc_);
    desc_ = nullptr;
  }
}

Status CudnnLRNDescriptor::Set(uint32_t N, double alpha, double beta, double K) {
  if (!desc_)
    CUDNN_RETURN_IF_ERROR(cudnnCreateLRNDescriptor(&desc_));

  CUDNN_RETURN_IF_ERROR(cudnnSetLRNDescriptor(desc_, N, alpha, beta, K));
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
