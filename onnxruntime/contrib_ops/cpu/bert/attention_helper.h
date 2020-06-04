// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/mlas/inc/mlas.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

template <typename T>
void ComputeAttentionSoftmaxInplace(T* score, int N, int D, ThreadPool* tp) {
  ThreadPool::TryParallelFor(tp, N, D * 2.0, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    for (std::ptrdiff_t j = begin; j != end; ++j) {
      float* x = reinterpret_cast<T*>(score) + j * D;
      float* y = x;

      // e^x is represented as infinity if x is large enough, like 100.f.
      // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if
      // one or more item are large enough. a math transform as below is
      // leveraged to get a stable softmax: e^xi/(e^x1 + ...e^xn) = e^(xi -
      // max) / (e^(x1 - max) + ... + e^(xn - max))
      float max = -std::numeric_limits<float>::infinity();
      for (int i = 0; i < D; i++) {
        if (max < x[i])
          max = x[i];
      }
      for (int i = 0; i < D; i++) {
        y[i] = expf(x[i] - max);
      }

      double sum = 0.0;

      for (int i = 0; i < D; i++) {
        sum += x[i];
      }

      if (sum == 0) {
        for (int i = 0; i < D; i++) {
          y[i] = 1.0f / (float)D;
        }
      } else {
        for (int i = 0; i < D; i++) {
          y[i] = x[i] / (float)sum;
        }
      }
    }
  });
}

template <>
inline void ComputeAttentionSoftmaxInplace(float* score, int N, int D, ThreadPool* tp) {
  MlasComputeSoftmax(score, score, N, D, false, tp);
}

template <typename T>
void PrepareMask(const int32_t* mask_index,
                 T* mask_data,
                 bool is_unidirectional,
                 int batch_size,
                 int sequence_length,
                 int past_sequence_length) {
  const int all_sequence_length = past_sequence_length + sequence_length;
  T* p_mask = mask_data;
  if (is_unidirectional) {  
    // unidirectional mask has shape SxS*
    for (int s_i = 0; s_i < sequence_length - 1; s_i++) {
      for (int m_i = past_sequence_length + s_i + 1; m_i < all_sequence_length; m_i++) {
        p_mask[s_i * all_sequence_length + m_i] = static_cast<T>(-10000.0);
      }
    }
    return;
  }

  ORT_ENFORCE(mask_index, "mask index should not be null.");
  for (int b_i = 0; b_i < batch_size; b_i++) {
    // TODO: mask_index can be used in softmax to save some calculation.
    // Convert mask_index to mask (-10000 means out of range, which will be 0 after softmax): B => BxS*
    int valid_length = mask_index[b_i];
    for (int m_i = valid_length; m_i < all_sequence_length; m_i++) {
      p_mask[m_i] = static_cast<T>(-10000.0);
    }

    // Broadcast mask from BxS* to BxSxS*
    for (int s_i = 1; s_i < sequence_length; s_i++) {
      memcpy(p_mask + s_i * all_sequence_length, p_mask, all_sequence_length * sizeof(T));
    }
    p_mask += sequence_length * sequence_length;
  }
}

// Concatenate a past state chunk S'xH with input state chunk SxH into present state chunk S*xH
// Returns a pointer to the start of present state chunk.
template<typename T>
T* ConcatStateChunk(const T* past, const T* chunk, T* present, size_t past_chunk_length, size_t present_chunk_length, std::ptrdiff_t i) {
  T* start = present + i * present_chunk_length;

  T* p = start;
  if (nullptr != past) {
    const T* src_past = past + i * past_chunk_length;
    memcpy(p, src_past, past_chunk_length * sizeof(T));
    p += past_chunk_length;
  }

  memcpy(p, chunk, (present_chunk_length - past_chunk_length) * sizeof(T));
  return start;
}

// Helper function to compute the attention probs. It does 2 things:
//  I. attention_probs(B, N, S, S*) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, S*, H -> B, N, H, S*) +
//                                    1 x mask_data(B, N, S, S*)
//  II.attention_probs(B, N, S, S*) = Softmax(attention_probs)
template <typename T>
void ComputeAttentionProbs(T* attention_probs,         // output buffer for the attention probs. Its size is BxNxSxS
                           const T* Q,                 // Q data. Its size is BxNxSxH
                           const T* K,                 // k data. Its size is BxNxSxH
                           const int32_t* mask_index,  // mask index. nullptr if no mask or its size is B
                           T* mask_data,               // buffer for mask data. Its size is: SxS* if is_unidirectional; BxSxS* if mask_index; null otherwise
                           int batch_size,             // batch size of self-attention
                           int sequence_length,        // sequence length of self-attention
                           int past_sequence_length,   // sequence length of past state
                           int head_size,              // head size of self-attention
                           int num_heads,              // number of heads of self-attention
                           bool is_unidirectional,     // indicate if it is unidrectional.
                           const T* past,              // past state
                           T* present,                 // present state
                           ThreadPool* tp) {
  const int all_sequence_length = past_sequence_length + sequence_length;                  // S* = S' + S
  const size_t past_chunk_length = static_cast<size_t>(past_sequence_length * head_size);  // S' x H
  const size_t input_chunk_length = static_cast<size_t>(sequence_length * head_size);      // S x H
  const size_t present_chunk_length = past_chunk_length + input_chunk_length;              // S* x H

  {
    if (mask_data != nullptr) {
      PrepareMask(mask_index, mask_data, is_unidirectional, batch_size, sequence_length, past_sequence_length);
    } else {  // no any mask
      memset(attention_probs, 0, batch_size * num_heads * sequence_length * all_sequence_length * sizeof(T));
    }

    const int loop_len = batch_size * num_heads;
    const float alpha = 1.0f / sqrt(static_cast<float>(head_size));

    // The cost of Gemm
    const double cost = static_cast<double>(head_size * sequence_length * all_sequence_length);

    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const std::ptrdiff_t batch_index = i / num_heads;

        // broadcast mask data: SxS* or (Bx)SxS* -> (BxNx)SxS*
        if (mask_data != nullptr) {
          const T* broadcast_data_src = is_unidirectional ? reinterpret_cast<T*>(mask_data) : reinterpret_cast<T*>(mask_data) + batch_index * sequence_length * all_sequence_length;
          T* broadcast_data_dest = reinterpret_cast<T*>(attention_probs) + sequence_length * all_sequence_length * i;
          memcpy(broadcast_data_dest, broadcast_data_src, sequence_length * all_sequence_length * sizeof(T));
        }

        const T* k = K + input_chunk_length * i;
        if (nullptr != present) {
          // concatenate past_K and K : (BxNx)S'xH, (BxNx)SxH -> (BxNx)S*xH
          k = ConcatStateChunk(past, k, present, past_chunk_length, present_chunk_length, i);
        }

        // gemm
        //                     original                 transposed             each iteration
        // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
        // B: K'               (B x N x) S* x H         (B x N x) H x S*       H x S*
        // C: attention_probs  (B x N x) S x S*         (B x N x) S x S*       S x S*
        math::Gemm<T, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, all_sequence_length, head_size, alpha,
                                  Q + input_chunk_length * i, k, 1.0,
                                  reinterpret_cast<T*>(attention_probs) + sequence_length * all_sequence_length * i, nullptr);
      }
    });
  }

  //  attention_probs(B, N, S, S*) = Softmax(attention_probs)
  {
    const int N = batch_size * num_heads * sequence_length;
    const int D = all_sequence_length;
    ComputeAttentionSoftmaxInplace(attention_probs, N, D, tp);
  }
}

template <typename T>
void ComputeVxAttentionScore(T* output,                 // buffer for the result with size BxSxNxH
                             T* tmp_buffer,             // buffer for temp use with size is BxNxSxH
                             const T* attention_probs,  // Attention probs with size BxNxSxS*
                             const T* V,                // V value with size BxNxSxH
                             int batch_size,            // batch size
                             int sequence_length,       // sequence length
                             int past_sequence_length,  // sequence length in past state
                             int head_size,             // head size
                             int num_heads,             // number of heads
                             int hidden_size,           // hidden size
                             const T* past,             // past state
                             T* present,                // present state
                             ThreadPool* tp) {
  const int all_sequence_length = past_sequence_length + sequence_length;                  // S* = S' + S
  const size_t past_chunk_length = static_cast<size_t>(past_sequence_length * head_size);  // S' x H
  const size_t input_chunk_length = static_cast<size_t>(sequence_length * head_size);      // S x H
  const size_t present_chunk_length = past_chunk_length + input_chunk_length;              // S* x H

  // Move the pointer of past and present to start of v values.
  if (nullptr != past) {
    past += batch_size * num_heads * past_sequence_length * head_size;
  }
  if (nullptr != present) {
    present += batch_size * num_heads * all_sequence_length * head_size;
  }

  const double cost =
      static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(sequence_length);

  ThreadPool::TryParallelFor(tp, batch_size * num_heads, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    for (std::ptrdiff_t i = begin; i != end; ++i) {
      
      const T* v = V + input_chunk_length * i;
      if (nullptr != present) {
        // concatenate past_V and V: (BxNx)S'xH, (BxNx)SxH -> (BxNx)S*xH
        v = ConcatStateChunk(past, v, present, past_chunk_length, present_chunk_length, i);
      }

      T* current_tmp_data = reinterpret_cast<T*>(tmp_buffer) + input_chunk_length * i;
      math::MatMul<T>(sequence_length, head_size, all_sequence_length,
                      attention_probs + sequence_length * all_sequence_length * i,
                      v, current_tmp_data, nullptr);

      // transpose: out(B, S, N, H) = transpose out_tmp(B, N, S, H)
      const int batch_index = static_cast<int>(i / num_heads);
      const int head_index = static_cast<int>(i % num_heads);
      T* src = current_tmp_data;
      T* dest = output + (batch_index * sequence_length * num_heads + head_index) * head_size;
      const auto bytes_to_copy = SafeInt<size_t>(head_size) * sizeof(T);
      for (int j = 0; j < sequence_length; j++) {
        memcpy(dest, src, bytes_to_copy);
        src += head_size;
        dest += hidden_size;
      }
    }
  });
}

}  // namespace contrib
}  // namespace onnxruntime
