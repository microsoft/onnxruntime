// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mixed_precision_scale.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define REGISTER_MIXEDPRECISIONSCALE_KERNEL_TYPED(SrcT)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      MixedPrecisionScale,                                                      \
      kOnnxDomain,                                                              \
      9,                                                                        \
      SrcT,                                                                     \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder()                                                        \
          .TypeConstraint("SrcT", DataTypeImpl::GetTensorType<SrcT>())          \
          .TypeConstraint("ScaleT", DataTypeImpl::GetTensorType<float>())       \
          .TypeConstraint("DstT", DataTypeImpl::AllFloatingPointTensorTypes()), \
      MixedPrecisionScale<SrcT>);

template <typename SrcT>
Status MixedPrecisionScale<SrcT>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<SrcT>::MappedType CudaSrcT;

  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* scale = context->Input<Tensor>(1);
  const CudaSrcT* x_data = reinterpret_cast<const CudaSrcT*>(X->template Data<SrcT>());
  const float* scale_data = scale->template Data<float>();

  const TensorShape& shape = X->Shape();
  Tensor* Y = context->Output(0, shape);
  size_t count = shape.Size();

#define CASE(TP_TYPE, DstT)                                                                        \
  case TP_TYPE:                                                                                    \
    Impl_MixedPrecisionScale<CudaSrcT, typename ToCudaType<DstT>::MappedType>(                     \
        x_data,                                                                                    \
        scale_data,                                                                                \
        reinterpret_cast<typename ToCudaType<DstT>::MappedType*>(Y->template MutableData<DstT>()), \
        count);                                                                                    \
    break;

  switch (to_) {
    CASE(TensorProto_DataType_FLOAT16, MLFloat16)
    CASE(TensorProto_DataType_FLOAT, float)
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected 'to' argument value: ", to_);
  }

  return Status::OK();
}

REGISTER_MIXEDPRECISIONSCALE_KERNEL_TYPED(MLFloat16)
REGISTER_MIXEDPRECISIONSCALE_KERNEL_TYPED(float)

template Status MixedPrecisionScale<MLFloat16>::ComputeInternal(OpKernelContext* context) const;
template Status MixedPrecisionScale<float>::ComputeInternal(OpKernelContext* context) const;

}  // namespace cuda
}  // namespace onnxruntime
