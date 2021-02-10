// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mixed_precision_scale.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#define ALL_IEEE_FLOAT_TENSOR_TYPES {DataTypeImpl::GetTensorType<float>(),      \
                                     DataTypeImpl::GetTensorType<double>(),     \
                                     DataTypeImpl::GetTensorType<MLFloat16>(),  \
                                     DataTypeImpl::GetTensorType<BFloat16>()}
#else
#define ALL_IEEE_FLOAT_TENSOR_TYPES DataTypeImpl::AllIEEEFloatTensorTypes()
#endif

#define REGISTER_MIXEDPRECISIONSCALE_KERNEL_TYPED(SrcT)                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      MixedPrecisionScale,                                                  \
      kMSDomain,                                                            \
      1,                                                                    \
      SrcT,                                                                 \
      kCudaExecutionProvider,                                               \
      KernelDefBuilder()                                                    \
          .TypeConstraint("SrcT", DataTypeImpl::GetTensorType<SrcT>())      \
          .TypeConstraint("ScaleT", DataTypeImpl::GetTensorType<float>())   \
          .TypeConstraint("DstT", ALL_IEEE_FLOAT_TENSOR_TYPES),             \
      MixedPrecisionScale<SrcT>);

Status BytesPerElement(ONNX_NAMESPACE::TensorProto_DataType to, size_t& bytes_per_elem) {
  switch (to) {
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      bytes_per_elem = sizeof(double);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      bytes_per_elem = sizeof(float);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      bytes_per_elem = sizeof(MLFloat16);
      break;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      bytes_per_elem = sizeof(BFloat16);
      break;
#endif
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected 'to' argument value: ", to);
  }
  return Status::OK();
}

template <typename SrcT>
MixedPrecisionScale<SrcT>::MixedPrecisionScale(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t to;
  Status status = info.GetAttr("to", &to);
  ORT_ENFORCE(status.IsOK(), "Attribute to is not set.");
  to_ = gsl::narrow_cast<ONNX_NAMESPACE::TensorProto_DataType>(to);

  status = BytesPerElement(to_, bytes_per_output_elem_);
  ORT_ENFORCE(status.IsOK(), status.ErrorMessage());

  int64_t fuse_outputs;
  info.GetAttrOrDefault("fuse_outputs", &fuse_outputs, static_cast<int64_t>(0));
  fuse_outputs_ = (fuse_outputs != 0);
}

template <typename SrcT>
Status MixedPrecisionScale<SrcT>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<SrcT>::MappedType CudaSrcT;

  const Tensor* scale = context->Input<Tensor>(0);
  const float* scale_data = scale->template Data<float>();

  // prepare outputs
  int num_inputs = context->InputCount() - 1;
  std::vector<void*> y_datas(num_inputs);
  if (fuse_outputs_) {
    int64_t total_num_elems = 0;
    std::vector<size_t> y_byte_offsets(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      const Tensor* X = context->Input<Tensor>(i + 1);
      y_byte_offsets[i] = total_num_elems * bytes_per_output_elem_;
      total_num_elems += X->Shape().Size();
    }

    Tensor* Y = context->Output(0, {total_num_elems});
    void* y_data = Y->MutableDataRaw();
    for (int i = 0; i < num_inputs; ++i) {
      y_datas[i] = (int8_t*)y_data + y_byte_offsets[i];
    }
  } else {
    for (int i = 0; i < num_inputs; ++i) {
      const Tensor* X = context->Input<Tensor>(i + 1);
      Tensor* Y = context->Output(i, X->Shape());
      y_datas[i] = Y->MutableDataRaw();
    }
  }

#define CASE(TP_TYPE, DstT)                                                    \
  case TP_TYPE:                                                                \
    Impl_MixedPrecisionScale<CudaSrcT, typename ToCudaType<DstT>::MappedType>( \
        Stream(),                                                              \
        x_data,                                                                \
        scale_data,                                                            \
        reinterpret_cast<typename ToCudaType<DstT>::MappedType*>(y_data),      \
        count);                                                                \
    break;

  for (int i = 0; i < num_inputs; ++i) {
    const Tensor* X = context->Input<Tensor>(i + 1);
    size_t count = X->Shape().Size();
    const CudaSrcT* x_data = reinterpret_cast<const CudaSrcT*>(X->template Data<SrcT>());
    auto y_data = y_datas[i];

    switch (to_) {
      CASE(TensorProto_DataType_FLOAT16, MLFloat16)
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
      CASE(TensorProto_DataType_BFLOAT16, BFloat16)
#endif
      CASE(TensorProto_DataType_FLOAT, float)
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unexpected 'to' argument value: ", to_);
    }
  }

  return Status::OK();
}

REGISTER_MIXEDPRECISIONSCALE_KERNEL_TYPED(MLFloat16)
REGISTER_MIXEDPRECISIONSCALE_KERNEL_TYPED(float)

template Status MixedPrecisionScale<MLFloat16>::ComputeInternal(OpKernelContext* context) const;
template Status MixedPrecisionScale<float>::ComputeInternal(OpKernelContext* context) const;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
REGISTER_MIXEDPRECISIONSCALE_KERNEL_TYPED(BFloat16)
template Status MixedPrecisionScale<BFloat16>::ComputeInternal(OpKernelContext* context) const;
#endif

}  // namespace cuda
}  // namespace onnxruntime
