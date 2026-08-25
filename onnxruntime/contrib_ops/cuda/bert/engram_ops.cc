// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/engram_ops.h"
#include "contrib_ops/cuda/bert/engram_ops_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_FLOAT_KERNEL_TYPED(Op, T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      Op,                                                          \
      kMSDomain,                                                   \
      1,                                                           \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),  \
      Op<T>);

REGISTER_FLOAT_KERNEL_TYPED(ShortConv, float)
REGISTER_FLOAT_KERNEL_TYPED(ShortConv, MLFloat16)
REGISTER_FLOAT_KERNEL_TYPED(ShortConv, BFloat16)
REGISTER_FLOAT_KERNEL_TYPED(EngramGate, float)
REGISTER_FLOAT_KERNEL_TYPED(EngramGate, MLFloat16)
REGISTER_FLOAT_KERNEL_TYPED(EngramGate, BFloat16)

#undef REGISTER_FLOAT_KERNEL_TYPED

#define REGISTER_INT_KERNEL_TYPED(T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      NgramHashMapping,                                           \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<T>()), \
      NgramHashMapping<T>);

REGISTER_INT_KERNEL_TYPED(int32_t)
REGISTER_INT_KERNEL_TYPED(int64_t)

#undef REGISTER_INT_KERNEL_TYPED

template <typename T>
ShortConv<T>::ShortConv(const OpKernelInfo& info) : CudaKernel(info) {
  activation_ = info.GetAttrOrDefault<std::string>("activation", "silu");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");
  dilation_ = info.GetAttrOrDefault<int64_t>("dilation", 1);
  ORT_ENFORCE(dilation_ >= 1, "dilation must be >= 1");
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

template <typename T>
Status ShortConv<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weight = context->Input<Tensor>(1);
  const Tensor* norm_scale = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);

  const TensorShape& input_shape = input->Shape();
  const TensorShape& weight_shape = weight->Shape();
  const TensorShape& scale_shape = norm_scale->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 4,
                    "input must have shape (batch_size, sequence_length, hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 3,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  ORT_RETURN_IF_NOT(scale_shape.NumDimensions() == 2,
                    "norm_scale must have shape (hc_mult, hidden_size)");

  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  const int64_t hc_mult = input_shape[2];
  const int64_t hidden_size = input_shape[3];
  const int64_t channels = hc_mult * hidden_size;
  const int64_t kernel_size = weight_shape[2];
  ORT_RETURN_IF_NOT(scale_shape[0] == hc_mult && scale_shape[1] == hidden_size,
                    "norm_scale shape must match input hc_mult and hidden_size");
  ORT_RETURN_IF_NOT(weight_shape[0] == channels && weight_shape[1] == 1,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  if (bias != nullptr) {
    ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 1 && bias->Shape()[0] == channels,
                      "bias must have shape (hc_mult * hidden_size)");
  }

  Tensor* output = context->Output(0, input_shape);
  return LaunchShortConvKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(input->Data<T>()),
      reinterpret_cast<const CudaT*>(weight->Data<T>()),
      reinterpret_cast<const CudaT*>(norm_scale->Data<T>()),
      bias == nullptr ? nullptr : reinterpret_cast<const CudaT*>(bias->Data<T>()),
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      batch_size,
      sequence_length,
      hc_mult,
      hidden_size,
      kernel_size,
      dilation_,
      epsilon_,
      activation_ == "silu" || activation_ == "swish");
}

template <typename T>
NgramHashMapping<T>::NgramHashMapping(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("max_ngram_size", &max_ngram_size_).IsOK(),
              "max_ngram_size attribute is required");
  ORT_ENFORCE(info.GetAttr<int64_t>("n_head_per_ngram", &n_head_per_ngram_).IsOK(),
              "n_head_per_ngram attribute is required");
  int64_t pad_id = 0;
  ORT_ENFORCE(info.GetAttr<int64_t>("pad_id", &pad_id).IsOK(), "pad_id attribute is required");
  ORT_ENFORCE(max_ngram_size_ >= 2, "max_ngram_size must be at least 2");
  ORT_ENFORCE(n_head_per_ngram_ >= 1, "n_head_per_ngram must be positive");
  ORT_ENFORCE(pad_id >= static_cast<int64_t>(std::numeric_limits<T>::min()) &&
                  pad_id <= static_cast<int64_t>(std::numeric_limits<T>::max()),
              "pad_id is out of range for the input id type");
  pad_id_ = static_cast<T>(pad_id);
}

template <typename T>
Status NgramHashMapping<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* multipliers = context->Input<Tensor>(1);
  const Tensor* vocab_sizes = context->Input<Tensor>(2);
  const TensorShape& input_shape = input_ids->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2, "input_ids must have rank 2");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 &&
                        multipliers->Shape()[0] >= max_ngram_size_,
                    "multipliers must have shape (max_ngram_size)");
  const int64_t num_heads = (max_ngram_size_ - 1) * n_head_per_ngram_;
  ORT_RETURN_IF_NOT(vocab_sizes->Shape().NumDimensions() == 1 && vocab_sizes->Shape()[0] == num_heads,
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");

  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  Tensor* output = context->Output(0, TensorShape({batch_size, sequence_length, num_heads}));
  return LaunchNgramHashMappingKernel<T>(
      Stream(context),
      input_ids->Data<T>(),
      multipliers->Data<T>(),
      vocab_sizes->Data<T>(),
      output->MutableData<T>(),
      batch_size,
      sequence_length,
      max_ngram_size_,
      n_head_per_ngram_,
      pad_id_);
}

template <typename T>
EngramGate<T>::EngramGate(const OpKernelInfo& info) : CudaKernel(info) {
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

template <typename T>
Status EngramGate<T>::ComputeInternal(OpKernelContext* context) const {
  using CudaT = typename OrtToCudaType<T>::type;
  const Tensor* embeddings = context->Input<Tensor>(0);
  const Tensor* hidden_states = context->Input<Tensor>(1);
  const Tensor* key_weight = context->Input<Tensor>(2);
  const Tensor* key_bias = context->Input<Tensor>(3);
  const Tensor* value_weight = context->Input<Tensor>(4);
  const Tensor* value_bias = context->Input<Tensor>(5);
  const Tensor* key_norm_scale = context->Input<Tensor>(6);
  const Tensor* query_norm_scale = context->Input<Tensor>(7);

  const TensorShape& embeddings_shape = embeddings->Shape();
  const TensorShape& hidden_shape = hidden_states->Shape();
  ORT_RETURN_IF_NOT(embeddings_shape.NumDimensions() == 3,
                    "embeddings must have shape (batch_size, sequence_length, embedding_size)");
  ORT_RETURN_IF_NOT(hidden_shape.NumDimensions() == 4,
                    "hidden_states must have shape (batch_size, sequence_length, hc_mult, hidden_size)");
  const int64_t batch_size = hidden_shape[0];
  const int64_t sequence_length = hidden_shape[1];
  const int64_t hc_mult = hidden_shape[2];
  const int64_t hidden_size = hidden_shape[3];
  const int64_t embedding_size = embeddings_shape[2];
  ORT_RETURN_IF_NOT(embeddings_shape[0] == batch_size && embeddings_shape[1] == sequence_length,
                    "embeddings and hidden_states batch/sequence dimensions must match");
  ORT_RETURN_IF_NOT(key_weight->Shape() == TensorShape({hc_mult, embedding_size, hidden_size}),
                    "key_weight must have shape (hc_mult, embedding_size, hidden_size)");
  ORT_RETURN_IF_NOT(value_weight->Shape() == TensorShape({embedding_size, hidden_size}),
                    "value_weight must have shape (embedding_size, hidden_size)");
  ORT_RETURN_IF_NOT(key_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "key_norm_scale must have shape (hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(query_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "query_norm_scale must have shape (hc_mult, hidden_size)");
  if (key_bias != nullptr) {
    ORT_RETURN_IF_NOT(key_bias->Shape() == TensorShape({hc_mult, hidden_size}),
                      "key_bias must have shape (hc_mult, hidden_size)");
  }
  if (value_bias != nullptr) {
    ORT_RETURN_IF_NOT(value_bias->Shape() == TensorShape({hidden_size}),
                      "value_bias must have shape (hidden_size)");
  }

  Tensor* output = context->Output(0, hidden_shape);
  return LaunchEngramGateKernel<CudaT>(
      Stream(context),
      reinterpret_cast<const CudaT*>(embeddings->Data<T>()),
      reinterpret_cast<const CudaT*>(hidden_states->Data<T>()),
      reinterpret_cast<const CudaT*>(key_weight->Data<T>()),
      key_bias == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key_bias->Data<T>()),
      reinterpret_cast<const CudaT*>(value_weight->Data<T>()),
      value_bias == nullptr ? nullptr : reinterpret_cast<const CudaT*>(value_bias->Data<T>()),
      reinterpret_cast<const CudaT*>(key_norm_scale->Data<T>()),
      reinterpret_cast<const CudaT*>(query_norm_scale->Data<T>()),
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      batch_size,
      sequence_length,
      hc_mult,
      hidden_size,
      embedding_size,
      epsilon_);
}

template class ShortConv<float>;
template class ShortConv<MLFloat16>;
template class ShortConv<BFloat16>;
template class NgramHashMapping<int32_t>;
template class NgramHashMapping<int64_t>;
template class EngramGate<float>;
template class EngramGate<MLFloat16>;
template class EngramGate<BFloat16>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
