// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class AttentionBase {
 protected:
  AttentionBase(const OpKernelInfo& info);
  Status CheckInputs(const Tensor* input,
                     const Tensor* weights,
                     const Tensor* bias,
                     const Tensor* mask_index,
                     const Tensor* past) const;

  Tensor* GetPresent(OpKernelContext* context,
                     const Tensor* past,
                     int batch_size,
                     int head_size,
                     int sequence_length,
                     int& past_sequence_length) const;

  template <typename T>
  Status ApplyAttention(const T* Q,                // Q data. Its size is BxNxSxH
                        const T* K,                // K data. Its size is BxNxSxH
                        const T* V,                // V value with size BxNxSxH
                        const Tensor* mask_index,  // mask index. nullptr if no mask or its size is B
                        const Tensor* past,        // past state
                        Tensor* output,            // output tensor
                        int batch_size,            // batch size
                        int sequence_length,       // sequence length
                        int head_size,             // head size
                        int hidden_size,           // hidden size
                        OpKernelContext* context) const {
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

    auto* tp = context->GetOperatorThreadPool();

    int past_sequence_length = 0;
    Tensor* present = GetPresent(context, past, batch_size, head_size, sequence_length, past_sequence_length);

    // Total sequence length including that of past state: S* = S' + S
    const int all_sequence_length = past_sequence_length + sequence_length;

    // Compute the attention score. It does 2 things:
    //         I. attention_probs(B, N, S, S*) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, S*, H -> B, N, H, S*) +
    //                                         1 x mask_data(B, N, S, S*)
    //         II.attention_probs(B, N, S, S*) = Softmax(attention_probs)
    size_t attention_probs_bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * all_sequence_length * sizeof(T);
    auto attention_probs = allocator->Alloc(attention_probs_bytes);
    BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));

    size_t mask_data_bytes = 0;
    if (mask_index != nullptr) {
      mask_data_bytes = SafeInt<size_t>(batch_size) * sequence_length * all_sequence_length * sizeof(T);
    } else if (is_unidirectional_) {
      mask_data_bytes = SafeInt<size_t>(sequence_length) * all_sequence_length * sizeof(T);
    }

    void* mask_data = nullptr;
    if (mask_data_bytes > 0) {
      mask_data = allocator->Alloc(mask_data_bytes);
      memset(mask_data, 0, mask_data_bytes);
    }
    BufferUniquePtr mask_data_buffer(mask_data, BufferDeleter(allocator));

    const int32_t* mask_index_data = mask_index != nullptr ? mask_index->template Data<int32_t>() : nullptr;
    const T* past_data = past != nullptr ? past->template Data<T>() : nullptr;
    T* present_data = present != nullptr ? present->template MutableData<T>() : nullptr;

    ComputeAttentionProbs<T>(static_cast<T*>(attention_probs), Q, K, mask_index_data, static_cast<T*>(mask_data),
                             batch_size, sequence_length, past_sequence_length, head_size, num_heads_, is_unidirectional_,
                             past_data, present_data, tp);

    // STEP.3: compute the attentionScore * Value. It does: out_tmp(B, N, S, H) = attention_probs(B, N, S, S*) x V(B, N, S*, H)
    auto out_tmp_data =
        allocator->Alloc(SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * head_size * sizeof(T));
    BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(allocator));

    ComputeVxAttentionScore(output->template MutableData<T>(), static_cast<T*>(out_tmp_data), static_cast<T*>(attention_probs), V,
                            batch_size, sequence_length, past_sequence_length, head_size, num_heads_, hidden_size,
                            past_data, present_data, tp);

    return Status::OK();
  }

  int num_heads_;           // number of attention heads
  bool is_unidirectional_;  // whether every token can only attend to previous tokens.
};

template <typename T>
class Attention : public OpKernel, public AttentionBase {
 public:
  explicit Attention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
