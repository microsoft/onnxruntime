// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/varlen_ngram_hash_mapping.h"

#include <limits>

#include "contrib_ops/cpu/bert/causal_conv_with_state_helper.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      VarlenNGramHashMapping,                                           \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("L", DataTypeImpl::GetTensorType<int32_t>()), \
      VarlenNGramHashMapping<T>);

REGISTER_KERNEL_TYPED(int32_t)
REGISTER_KERNEL_TYPED(int64_t)
#undef REGISTER_KERNEL_TYPED

template <typename T>
VarlenNGramHashMapping<T>::VarlenNGramHashMapping(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("max_ngram_size", &max_ngram_size_).IsOK());
  ORT_ENFORCE(info.GetAttr<int64_t>("n_head_per_ngram", &n_head_per_ngram_).IsOK());
  int64_t pad_id;
  ORT_ENFORCE(info.GetAttr<int64_t>("pad_id", &pad_id).IsOK());
  ORT_ENFORCE(max_ngram_size_ >= 2 && n_head_per_ngram_ >= 1);
  ORT_ENFORCE(pad_id >= static_cast<int64_t>(std::numeric_limits<T>::min()) &&
              pad_id <= static_cast<int64_t>(std::numeric_limits<T>::max()));
  pad_id_ = static_cast<T>(pad_id);
  max_checkpoints_ = info.GetAttrOrDefault<int64_t>("max_checkpoints", 0);
  ORT_ENFORCE(max_checkpoints_ >= 0 && max_checkpoints_ <= kMaxStateWindow);
}

template <typename T>
Status VarlenNGramHashMapping<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* multipliers = context->Input<Tensor>(1);
  const Tensor* vocab_sizes = context->Input<Tensor>(2);
  const Tensor* cu_seqlens = context->Input<Tensor>(3);
  const Tensor* initial_ids = context->Input<Tensor>(4);
  ORT_RETURN_IF_NOT(input_ids && multipliers && vocab_sizes && cu_seqlens && initial_ids,
                    "all inputs are required");
  ORT_RETURN_IF_NOT(input_ids->Shape().NumDimensions() == 1,
                    "input_ids must have shape (total_tokens)");
  const int64_t total_tokens = input_ids->Shape()[0];
  const int64_t state_length = max_ngram_size_ - 1;
  const int64_t num_heads = state_length * n_head_per_ngram_;
  ORT_RETURN_IF_NOT(multipliers->Shape() == TensorShape({max_ngram_size_}),
                    "multipliers must have shape (max_ngram_size)");
  ORT_RETURN_IF_NOT(vocab_sizes->Shape() == TensorShape({num_heads}),
                    "vocab_sizes has an invalid shape");
  ORT_RETURN_IF_NOT(cu_seqlens->Shape().NumDimensions() == 1 && cu_seqlens->Shape()[0] >= 2,
                    "cumulative_sequence_length must have shape (batch_size + 1)");
  const int64_t batch_size = cu_seqlens->Shape()[0] - 1;
  ORT_RETURN_IF_NOT(initial_ids->Shape() == TensorShape({batch_size, state_length}),
                    "initial_ids must have shape (batch_size, max_ngram_size - 1)");
  ORT_RETURN_IF_NOT(total_tokens >= batch_size && total_tokens <= std::numeric_limits<int>::max() &&
                        batch_size <= std::numeric_limits<int>::max() &&
                        max_ngram_size_ <= std::numeric_limits<int>::max() &&
                        n_head_per_ngram_ <= std::numeric_limits<int>::max() &&
                        num_heads <= std::numeric_limits<int>::max(),
                    "dimensions are invalid or too large for the CUDA kernel");

  Tensor* hash_ids = context->Output(0, TensorShape({total_tokens, num_heads}));
  Tensor* final_ids = context->Output(1, TensorShape({batch_size, state_length}));
  Tensor* prefix_ids = context->Output(2, TensorShape({max_checkpoints_, batch_size, state_length}));
  return LaunchVarlenNGramHashMappingKernel<T>(
      Stream(context), input_ids->Data<T>(), multipliers->Data<T>(), vocab_sizes->Data<T>(),
      cu_seqlens->Data<int32_t>(), initial_ids->Data<T>(), hash_ids->MutableData<T>(),
      final_ids->MutableData<T>(), prefix_ids ? prefix_ids->MutableData<T>() : nullptr,
      static_cast<int>(batch_size), static_cast<int>(total_tokens), static_cast<int>(max_ngram_size_),
      static_cast<int>(n_head_per_ngram_), static_cast<int>(max_checkpoints_), pad_id_);
}

template class VarlenNGramHashMapping<int32_t>;
template class VarlenNGramHashMapping<int64_t>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
