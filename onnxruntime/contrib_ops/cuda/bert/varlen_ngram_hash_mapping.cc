// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/varlen_ngram_hash_mapping.h"
#include "contrib_ops/cuda/bert/varlen_ngram_hash_mapping_impl.h"
#include "contrib_ops/cpu/bert/engram_helper.h"
#include "core/providers/cuda/cuda_common.h"

#include <limits>

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
          .TypeConstraint("S", DataTypeImpl::GetTensorType<int32_t>()), \
      VarlenNGramHashMapping<T>);

REGISTER_KERNEL_TYPED(int32_t)
REGISTER_KERNEL_TYPED(int64_t)

#undef REGISTER_KERNEL_TYPED

template <typename T>
VarlenNGramHashMapping<T>::VarlenNGramHashMapping(const OpKernelInfo& info) : CudaKernel(info) {
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
Status VarlenNGramHashMapping<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* multipliers = context->Input<Tensor>(1);
  const Tensor* vocab_sizes = context->Input<Tensor>(2);
  const Tensor* cu_seqlens = context->Input<Tensor>(3);
  const Tensor* past_ids = context->Input<Tensor>(4);

  ORT_RETURN_IF_NOT(input_ids != nullptr, "input_ids is required");
  ORT_RETURN_IF_NOT(cu_seqlens != nullptr, "cumulative_sequence_length input is required");
  ORT_RETURN_IF_NOT(input_ids->Shape().NumDimensions() == 1, "input_ids must have rank 1 (total_tokens)");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 &&
                        multipliers->Shape()[0] == max_ngram_size_,
                    "multipliers must have shape (max_ngram_size)");
  int64_t num_heads = 0;
  ORT_RETURN_IF_NOT(onnxruntime::contrib::engram_helper::TryMultiplyDims(max_ngram_size_ - 1, n_head_per_ngram_, num_heads),
                    "VarlenNGramHashMapping: (max_ngram_size - 1) * n_head_per_ngram overflows int64_t");
  ORT_RETURN_IF_NOT(vocab_sizes->Shape().NumDimensions() == 1 && vocab_sizes->Shape()[0] == num_heads,
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");

  const auto& cu_seqlens_shape = cu_seqlens->Shape();
  ORT_RETURN_IF_NOT(cu_seqlens_shape.NumDimensions() == 1 && cu_seqlens_shape[0] >= 2,
                    "cumulative_sequence_length must have rank 1 with at least 2 elements");
  // batch_size = cu_seqlens.Shape()[0] - 1. This is a shape-only computation; the offset values
  // themselves stay on the device and are validated inside the kernel, per the varlen convention
  // used by VarlenCausalConvWithState.
  const int64_t batch_size = cu_seqlens_shape[0] - 1;
  ORT_RETURN_IF_NOT(batch_size <= std::numeric_limits<int>::max(), "batch size is too large for the CUDA kernel");

  const int64_t total_tokens = input_ids->Shape()[0];
  ORT_RETURN_IF_NOT(total_tokens >= batch_size,
                    "total_tokens must be at least batch_size because every request must contain a token");

  const int64_t state_length = max_ngram_size_ - 1;
  if (past_ids != nullptr) {
    ORT_RETURN_IF_NOT(past_ids->Shape() == TensorShape({batch_size, state_length}),
                      "past_ids must have shape (batch_size, max_ngram_size - 1)");
  }

  Tensor* output = context->Output(0, TensorShape({total_tokens, num_heads}));
  Tensor* present_ids = context->Output(1, TensorShape({batch_size, state_length}));

  // Device-resident scratch flag: cumulative_sequence_length's values cannot be validated
  // host-side, so LaunchVarlenNGramHashMappingKernel computes global monotonicity into this flag
  // on-device before any output-producing kernel runs (see the impl file for why per-block local
  // checks alone are not sufficient).
  auto is_valid_buffer = GetScratchBuffer<int32_t>(1, GetComputeStream(context));

  return LaunchVarlenNGramHashMappingKernel<T>(
      Stream(context),
      input_ids->Data<T>(),
      multipliers->Data<T>(),
      vocab_sizes->Data<T>(),
      cu_seqlens->Data<int32_t>(),
      past_ids == nullptr ? nullptr : past_ids->Data<T>(),
      output->MutableData<T>(),
      present_ids == nullptr ? nullptr : present_ids->MutableData<T>(),
      batch_size,
      total_tokens,
      max_ngram_size_,
      n_head_per_ngram_,
      pad_id_,
      GetDeviceProp().maxThreadsPerBlock,
      is_valid_buffer.get());
}

template class VarlenNGramHashMapping<int32_t>;
template class VarlenNGramHashMapping<int64_t>;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
