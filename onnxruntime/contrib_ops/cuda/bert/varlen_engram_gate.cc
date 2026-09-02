// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/varlen_engram_gate.h"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      VarlenEngramGate,                                                 \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()), \
      VarlenEngramGate<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

#undef REGISTER_KERNEL_TYPED

template <typename T>
VarlenEngramGate<T>::VarlenEngramGate(const OpKernelInfo& info) : CudaKernel(info) {
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

template <typename T>
Status VarlenEngramGate<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;

  const Tensor* key = context->Input<Tensor>(0);
  const Tensor* query = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* key_norm_scale = context->Input<Tensor>(3);
  const Tensor* query_norm_scale = context->Input<Tensor>(4);
  const Tensor* cu_seqlens = context->Input<Tensor>(5);

  ORT_RETURN_IF_NOT(key != nullptr && query != nullptr && value != nullptr &&
                        key_norm_scale != nullptr && query_norm_scale != nullptr && cu_seqlens != nullptr,
                    "all inputs are required");

  const auto& key_shape = key->Shape();
  ORT_RETURN_IF_NOT(key_shape.NumDimensions() == 3,
                    "key must have shape (total_tokens, hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(query->Shape() == key_shape, "query must have the same shape as key");

  const int64_t total_tokens_64 = key_shape[0];
  const int64_t hc_mult_64 = key_shape[1];
  const int64_t hidden_size_64 = key_shape[2];
  ORT_RETURN_IF_NOT(total_tokens_64 <= std::numeric_limits<int>::max() &&
                        hc_mult_64 <= std::numeric_limits<int>::max() &&
                        hidden_size_64 <= std::numeric_limits<int>::max() &&
                        total_tokens_64 * hc_mult_64 <= std::numeric_limits<int>::max(),
                    "input dimensions are too large for the CUDA kernel");

  ORT_RETURN_IF_NOT(value->Shape() == TensorShape({total_tokens_64, hidden_size_64}),
                    "value must have shape (total_tokens, hidden_size)");
  ORT_RETURN_IF_NOT(key_norm_scale->Shape() == TensorShape({hc_mult_64, hidden_size_64}),
                    "key_norm_scale must have shape (hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(query_norm_scale->Shape() == TensorShape({hc_mult_64, hidden_size_64}),
                    "query_norm_scale must have shape (hc_mult, hidden_size)");

  const auto& cu_shape = cu_seqlens->Shape();
  ORT_RETURN_IF_NOT(cu_shape.NumDimensions() == 1 && cu_shape[0] >= 2,
                    "cumulative_sequence_length must have shape (batch_size + 1) with batch_size >= 1");
  const int64_t batch_size_64 = cu_shape[0] - 1;
  ORT_RETURN_IF_NOT(batch_size_64 <= std::numeric_limits<int>::max(),
                    "batch size is too large for the CUDA kernel");
  ORT_RETURN_IF_NOT(total_tokens_64 >= batch_size_64,
                    "total_tokens must be at least batch_size");

  Tensor* output = context->Output(0, key_shape);
  if (key_shape.Size() == 0) {
    return Status::OK();
  }

  return LaunchVarlenEngramGateKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(key->Data<T>()),
      reinterpret_cast<const CudaT*>(query->Data<T>()),
      reinterpret_cast<const CudaT*>(value->Data<T>()),
      reinterpret_cast<const CudaT*>(key_norm_scale->Data<T>()),
      reinterpret_cast<const CudaT*>(query_norm_scale->Data<T>()),
      cu_seqlens->Data<int32_t>(),
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      static_cast<int>(batch_size_64),
      static_cast<int>(total_tokens_64),
      static_cast<int>(hc_mult_64),
      static_cast<int>(hidden_size_64),
      epsilon_);
}

template class VarlenEngramGate<float>;
template class VarlenEngramGate<MLFloat16>;
template class VarlenEngramGate<BFloat16>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
