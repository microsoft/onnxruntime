// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/ngram_hash_mapping.h"
#include "contrib_ops/cuda/bert/ngram_hash_mapping_impl.h"
#include "core/providers/cuda/cuda_common.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      NGramHashMapping,                                           \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .MayInplace(3, 1)                                       \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<T>()), \
      NGramHashMapping<T>);

REGISTER_KERNEL_TYPED(int32_t)
REGISTER_KERNEL_TYPED(int64_t)

#undef REGISTER_KERNEL_TYPED

template <typename T>
NGramHashMapping<T>::NGramHashMapping(const OpKernelInfo& info) : CudaKernel(info) {
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
  reset_on_eos_ = info.GetAttrOrDefault<int64_t>("reset_on_eos", 0);
}

template <typename T>
Status NGramHashMapping<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* multipliers = context->Input<Tensor>(1);
  const Tensor* vocab_sizes = context->Input<Tensor>(2);
  const Tensor* past_ids = context->Input<Tensor>(3);
  const Tensor* head_offsets = context->Input<Tensor>(4);
  const Tensor* eos_token_id = context->Input<Tensor>(5);
  const Tensor* segment_ids = context->Input<Tensor>(6);
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
  const int64_t state_length = max_ngram_size_ - 1;
  if (past_ids != nullptr) {
    ORT_RETURN_IF_NOT(past_ids->Shape() == TensorShape({batch_size, state_length}),
                      "past_ids must have shape (batch_size, max_ngram_size - 1)");
  }
  if (head_offsets != nullptr) {
    ORT_RETURN_IF_NOT(head_offsets->Shape().NumDimensions() == 1 && head_offsets->Shape()[0] == num_heads,
                      "head_offsets must have shape ((max_ngram_size - 1) * n_head_per_ngram)");
  }
  if (eos_token_id != nullptr) {
    ORT_RETURN_IF_NOT(eos_token_id->Shape().Size() == 1, "eos_token_id must be a scalar");
  }
  if (segment_ids != nullptr) {
    ORT_RETURN_IF_NOT(segment_ids->Shape() == TensorShape({batch_size, sequence_length}),
                      "segment_ids must have shape (batch_size, sequence_length)");
  }

  Tensor* output = context->Output(0, TensorShape({batch_size, sequence_length, num_heads}));
  Tensor* present_ids = context->Output(1, TensorShape({batch_size, state_length}));
  return LaunchNGramHashMappingKernel<T>(
      Stream(context),
      input_ids->Data<T>(),
      multipliers->Data<T>(),
      vocab_sizes->Data<T>(),
      past_ids == nullptr ? nullptr : past_ids->Data<T>(),
      head_offsets == nullptr ? nullptr : head_offsets->Data<T>(),
      eos_token_id == nullptr ? nullptr : eos_token_id->Data<T>(),
      segment_ids == nullptr ? nullptr : segment_ids->Data<int32_t>(),
      output->MutableData<T>(),
      present_ids == nullptr ? nullptr : present_ids->MutableData<T>(),
      batch_size,
      sequence_length,
      max_ngram_size_,
      n_head_per_ngram_,
      pad_id_,
      reset_on_eos_ != 0);
}

template class NGramHashMapping<int32_t>;
template class NGramHashMapping<int64_t>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
