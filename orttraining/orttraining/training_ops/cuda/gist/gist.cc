// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gist.h"
#include "gist_impl.h"

namespace onnxruntime {
namespace cuda {

// Pack Binary
#define REGISTER_KERNEL_TYPED_BIN_ENC(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      GistBinarizeEncoder,                                                      \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GistBinarizeEncoderOp<T>);

REGISTER_KERNEL_TYPED_BIN_ENC(float)
REGISTER_KERNEL_TYPED_BIN_ENC(MLFloat16)
REGISTER_KERNEL_TYPED_BIN_ENC(double)

template <typename T>
Status GistBinarizeEncoderOp<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF(X == nullptr, "X input is unavailable");
  Tensor* Y = context->Output(0, X->Shape());
  typedef typename ToCudaType<T>::MappedType CudaT;

  GistBinarizeEncoderImpl<CudaT>(
      Stream(),
      reinterpret_cast<const CudaT*>(X->template Data<T>()),
      reinterpret_cast<bool*>(Y->template MutableData<bool>()),
      Y->Shape().Size());

  return Status::OK();
}

#define REGISTER_KERNEL_TYPED_BIN_DEC(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      GistBinarizeDecoder,                                                      \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GistBinarizeDecoderOp<T>);

REGISTER_KERNEL_TYPED_BIN_DEC(float)
REGISTER_KERNEL_TYPED_BIN_DEC(MLFloat16)
REGISTER_KERNEL_TYPED_BIN_DEC(double)

template <typename T>
Status GistBinarizeDecoderOp<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF(X == nullptr, "X input is unavailable");
  Tensor* Y = context->Output(0, X->Shape());
  typedef typename ToCudaType<T>::MappedType CudaT;

  GistBinarizeDecoderImpl<CudaT>(
      Stream(),
      reinterpret_cast<const bool*>(X->template Data<bool>()),
      reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
      Y->Shape().Size());

  return Status::OK();
}

// Pack1
#define REGISTER_KERNEL_TYPED_PACK1_ENC(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      GistPack1Encoder,                                                         \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GistPack1EncoderOp<T>);

REGISTER_KERNEL_TYPED_PACK1_ENC(bool)
REGISTER_KERNEL_TYPED_PACK1_ENC(float)

template <typename T>
Status GistPack1EncoderOp<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF(X == nullptr, "X input is unavailable");

  long n = static_cast<long>((X->Shape().Size() + GIST_PACK1_FACTOR - 1) / GIST_PACK1_FACTOR);
  Tensor* Y = context->Output(0, TensorShape({n}));
  typedef typename ToCudaType<T>::MappedType CudaT;
  GistPack1EncoderImpl<CudaT>(
      Stream(),
      reinterpret_cast<const CudaT*>(X->template Data<T>()),
      reinterpret_cast<uint8_t*>(Y->template MutableData<uint8_t>()),
      n);

  return Status::OK();
}

#define REGISTER_KERNEL_TYPED_PACK1_DEC(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      GistPack1Decoder,                                                         \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GistPack1DecoderOp<T>);

REGISTER_KERNEL_TYPED_PACK1_DEC(bool)
REGISTER_KERNEL_TYPED_PACK1_DEC(float)

template <typename T>
Status GistPack1DecoderOp<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF(X == nullptr, "X input is unavailable");

  Tensor* Y = context->Output(0, TensorShape({X->Shape().Size() * GIST_PACK1_FACTOR}));
  typedef typename ToCudaType<T>::MappedType CudaT;
  GistPack1DecoderImpl<CudaT>(
      Stream(),
      reinterpret_cast<const uint8_t*>(X->template Data<uint8_t>()),
      reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
      Y->Shape().Size());

  return Status::OK();
}

// Pack 8
#define REGISTER_KERNEL_TYPED_PACK8_ENC(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      GistPack8Encoder,                                                         \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GistPack8EncoderOp<T>);

REGISTER_KERNEL_TYPED_PACK8_ENC(float)
REGISTER_KERNEL_TYPED_PACK8_ENC(MLFloat16)

template <typename T>
Status GistPack8EncoderOp<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF(X == nullptr, "X input is unavailable");

  Tensor* Y = context->Output(0, X->Shape());

  typedef typename ToCudaType<T>::MappedType CudaT;

  GistPack8EncoderImpl<CudaT>(
      Stream(),
      reinterpret_cast<const CudaT*>(X->template Data<T>()),
      reinterpret_cast<uint8_t*>(Y->template MutableData<uint8_t>()),
      Y->Shape().Size());

  return Status::OK();
}

#define REGISTER_KERNEL_TYPED_PACK8_DEC(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      GistPack8Decoder,                                                         \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GistPack8DecoderOp<T>);

REGISTER_KERNEL_TYPED_PACK8_DEC(float)
REGISTER_KERNEL_TYPED_PACK8_DEC(MLFloat16)

template <typename T>
Status GistPack8DecoderOp<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF(X == nullptr, "X input is unavailable");
  Tensor* Y = context->Output(0, X->Shape());

  typedef typename ToCudaType<T>::MappedType CudaT;

  GistPack8DecoderImpl<CudaT>(
      Stream(),
      reinterpret_cast<const uint8_t*>(X->template Data<uint8_t>()),
      reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
      Y->Shape().Size());

  return Status::OK();
}

// Pack 16
#define REGISTER_KERNEL_TYPED_PACK16_ENC(T)                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      GistPack16Encoder,                                                        \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GistPack16EncoderOp<T>);

REGISTER_KERNEL_TYPED_PACK16_ENC(float)

template <typename T>
Status GistPack16EncoderOp<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF(X == nullptr, "X input is unavailable");

  Tensor* Y = context->Output(0, X->Shape());

  typedef typename ToCudaType<T>::MappedType CudaT;

  GistPack16EncoderImpl<CudaT>(
      Stream(),
      reinterpret_cast<const CudaT*>(X->template Data<T>()),
      reinterpret_cast<half*>(Y->template MutableData<MLFloat16>()),
      Y->Shape().Size());

  return Status::OK();
}

#define REGISTER_KERNEL_TYPED_PACK16_DEC(T)                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      GistPack16Decoder,                                                        \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GistPack16DecoderOp<T>);

REGISTER_KERNEL_TYPED_PACK16_DEC(float)

template <typename T>
Status GistPack16DecoderOp<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF(X == nullptr, "X input is unavailable");
  Tensor* Y = context->Output(0, X->Shape());

  typedef typename ToCudaType<T>::MappedType CudaT;

  GistPack16DecoderImpl<CudaT>(
      Stream(),
      reinterpret_cast<const half*>(X->template Data<MLFloat16>()),
      reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
      Y->Shape().Size());

  return Status::OK();
}

// Pack MSFP15
#define REGISTER_KERNEL_TYPED_PACKMSFP15_ENC(T)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      GistPackMsfp15Encoder,                                                    \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GistPackMsfp15EncoderOp<T>);

REGISTER_KERNEL_TYPED_PACKMSFP15_ENC(float)

template <typename T>
Status GistPackMsfp15EncoderOp<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF(X == nullptr, "X input is unavailable");

  Tensor* Y = context->Output(0, X->Shape());

  const auto shape = X->Shape();
  const size_t ndims = shape.NumDimensions();
  const size_t pre_axis_size = shape.SizeToDimension(ndims - 1);
  size_t axis_size = shape.SizeFromDimension(ndims - 1);
  const size_t tile_size = 8;
  if (axis_size % tile_size != 0)
    axis_size = shape.SizeToDimension(ndims - 2);

  typedef typename ToCudaType<T>::MappedType CudaT;

  GistPackMsfp15EncoderImpl<CudaT>(
      Stream(),
      reinterpret_cast<const CudaT*>(X->template Data<T>()),
      reinterpret_cast<uint8_t*>(Y->template MutableData<uint8_t>()),
      pre_axis_size,
      axis_size,
      tile_size);

  return Status::OK();
}

#define REGISTER_KERNEL_TYPED_PACKMSFP15_DEC(T)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      GistPackMsfp15Decoder,                                                    \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GistPackMsfp15DecoderOp<T>);

REGISTER_KERNEL_TYPED_PACKMSFP15_DEC(float)

template <typename T>
Status GistPackMsfp15DecoderOp<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_RETURN_IF(X == nullptr, "X input is unavailable");
  Tensor* Y = context->Output(0, X->Shape());

  const auto shape = X->Shape();
  const size_t ndims = shape.NumDimensions();
  const size_t pre_axis_size = shape.SizeToDimension(ndims - 1);
  size_t axis_size = shape.SizeFromDimension(ndims - 1);
  const size_t tile_size = 8;
  if (axis_size % tile_size != 0)
    axis_size = shape.SizeToDimension(ndims - 2);
  typedef typename ToCudaType<T>::MappedType CudaT;

  GistPackMsfp15DecoderImpl<CudaT>(
      Stream(),
      reinterpret_cast<const uint8_t*>(X->template Data<uint8_t>()),
      reinterpret_cast<CudaT*>(Y->template MutableData<T>()),
      pre_axis_size,
      axis_size,
      tile_size);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
