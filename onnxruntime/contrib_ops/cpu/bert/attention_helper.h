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
                 int mask_index_length,
                 T* mask_data,
                 bool is_unidirectional,
                 int batch_size,
                 int sequence_length,
                 int past_sequence_length) {
  const int all_sequence_length = past_sequence_length + sequence_length;

  // mask_data has been filled with 0, and its shape is BxSxS*
  T* p_mask = mask_data;

  for (int b_i = 0; b_i < batch_size; b_i++) {
    // TODO: mask_index can be used in softmax to save some calculation.

    // Convert mask_index to mask (-10000 means no attention, which will be 0 after softmax)
    if (nullptr != mask_index) {
      // mask_index is 1D: (B) or (2B) => (Bx)S*

      // Apply mask for right-side padding
      int end_position = mask_index[b_i];
      for (int m_i = end_position; m_i < all_sequence_length; m_i++) {
        p_mask[m_i] = static_cast<T>(-10000.0);
      }

      // Apply mask for left-side padding
      if (mask_index_length == 2 * batch_size) {
        int start_position = std::min(mask_index[b_i + batch_size], all_sequence_length);
        for (int m_i = 0; m_i < start_position; m_i++) {
          p_mask[m_i] = static_cast<T>(-10000.0);
        }
      }
    }

    // Broadcast mask from (Bx)S* to (Bx)SxS*
    for (int s_i = 1; s_i < sequence_length; s_i++) {
      memcpy(p_mask + s_i * all_sequence_length, p_mask, all_sequence_length * sizeof(T));
    }

    // Apply unidirectional mask.
    if (is_unidirectional) {
      for (int s_i = 0; s_i < sequence_length - 1; s_i++) {
        for (int m_i = past_sequence_length + s_i + 1; m_i < all_sequence_length; m_i++) {
          p_mask[s_i * all_sequence_length + m_i] = static_cast<T>(-10000.0);
        }
      }
    }

    p_mask += sequence_length * all_sequence_length;
  }
}

// Concatenate a past state chunk S'xH with input state chunk SxH into present state chunk S*xH
// Returns a pointer to the start of present state chunk.
template <typename T>
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

}  // namespace contrib
}  // namespace onnxruntime
