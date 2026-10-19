// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/engram_gate.h"
#include "contrib_ops/cuda/bert/engram_gate_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      EngramGate,                                                 \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      EngramGate<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#undef REGISTER_KERNEL_TYPED

template <typename T>
EngramGate<T>::EngramGate(const OpKernelInfo& info) : CudaKernel(info) {
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

template <typename T>
Status EngramGate<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;
  const Tensor* key = context->Input<Tensor>(0);
  const Tensor* query = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* key_norm_scale = context->Input<Tensor>(3);
  const Tensor* query_norm_scale = context->Input<Tensor>(4);
  const Tensor* conv_norm_scale = context->Input<Tensor>(5);

  const TensorShape& key_shape = key->Shape();
  ORT_RETURN_IF_NOT(key_shape.NumDimensions() == 4,
                    "key must have shape (batch_size, sequence_length, hc_mult, hidden_size)");
  const int64_t batch_size = key_shape[0];
  const int64_t sequence_length = key_shape[1];
  const int64_t hc_mult = key_shape[2];
  const int64_t hidden_size = key_shape[3];

  ORT_RETURN_IF_NOT(query->Shape() == key_shape, "query must have the same shape as key");
  ORT_RETURN_IF_NOT(value->Shape() == TensorShape({batch_size, sequence_length, hidden_size}),
                    "value must have shape (batch_size, sequence_length, hidden_size)");
  ORT_RETURN_IF_NOT(key_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "key_norm_scale must have shape (hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(query_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "query_norm_scale must have shape (hc_mult, hidden_size)");
  if (conv_norm_scale != nullptr) {
    ORT_RETURN_IF_NOT(conv_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                      "conv_norm_scale must have shape (hc_mult, hidden_size)");
  }

  Tensor* output = context->Output(0, key_shape);
  Tensor* output_normed = context->OutputCount() > 1 ? context->Output(1, key_shape) : nullptr;
  ORT_RETURN_IF_NOT(output_normed == nullptr || conv_norm_scale != nullptr,
                    "conv_norm_scale is required to produce the gated_value_normed output");
  if (key_shape.Size() == 0) {
    return Status::OK();
  }

  return LaunchEngramGateKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(key->Data<T>()),
      reinterpret_cast<const CudaT*>(query->Data<T>()),
      reinterpret_cast<const CudaT*>(value->Data<T>()),
      reinterpret_cast<const CudaT*>(key_norm_scale->Data<T>()),
      reinterpret_cast<const CudaT*>(query_norm_scale->Data<T>()),
      conv_norm_scale == nullptr ? nullptr : reinterpret_cast<const CudaT*>(conv_norm_scale->Data<T>()),
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      output_normed == nullptr ? nullptr : reinterpret_cast<CudaT*>(output_normed->MutableData<T>()),
      batch_size,
      sequence_length,
      hc_mult,
      hidden_size,
      epsilon_);
}

template class EngramGate<float>;
template class EngramGate<MLFloat16>;
template class EngramGate<BFloat16>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
