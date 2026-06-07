// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/llm/onnx_flash_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "core/common/safeint.h"
#include "core/providers/cpu/llm/attention.h"

using onnxruntime::attention_helper::AttentionParameters;
using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace {

const float* QueryPointer(const float* Q, const AttentionParameters& parameters,
                          int batch, int q_head, int q_index) {
  if (parameters.transpose_output) {
    const ptrdiff_t offset =
        (SafeInt<ptrdiff_t>(batch) * parameters.q_sequence_length * parameters.q_num_heads +
         SafeInt<ptrdiff_t>(q_index) * parameters.q_num_heads + q_head) *
        parameters.head_size;
    return Q + offset;
  }

  const ptrdiff_t offset =
      ((SafeInt<ptrdiff_t>(batch) * parameters.q_num_heads + q_head) *
           parameters.q_sequence_length +
       q_index) *
      parameters.head_size;
  return Q + offset;
}

const float* KeyPointer(const float* K, const AttentionParameters& parameters,
                        int batch, int kv_head, int kv_index) {
  if (parameters.transpose_output) {
    const ptrdiff_t offset =
        (SafeInt<ptrdiff_t>(batch) * parameters.kv_sequence_length * parameters.kv_num_heads +
         SafeInt<ptrdiff_t>(kv_index) * parameters.kv_num_heads + kv_head) *
        parameters.head_size;
    return K + offset;
  }

  const ptrdiff_t offset =
      ((SafeInt<ptrdiff_t>(batch) * parameters.kv_num_heads + kv_head) *
           parameters.kv_sequence_length +
       kv_index) *
      parameters.head_size;
  return K + offset;
}

const float* ValuePointer(const float* V, const AttentionParameters& parameters,
                          int batch, int kv_head, int kv_index) {
  if (parameters.transpose_output) {
    const ptrdiff_t offset =
        (SafeInt<ptrdiff_t>(batch) * parameters.kv_sequence_length * parameters.kv_num_heads +
         SafeInt<ptrdiff_t>(kv_index) * parameters.kv_num_heads + kv_head) *
        parameters.v_head_size;
    return V + offset;
  }

  const ptrdiff_t offset =
      ((SafeInt<ptrdiff_t>(batch) * parameters.kv_num_heads + kv_head) *
           parameters.kv_sequence_length +
       kv_index) *
      parameters.v_head_size;
  return V + offset;
}

float* OutputPointer(float* output, const AttentionParameters& parameters,
                     int batch, int q_head, int q_index) {
  if (parameters.transpose_output) {
    const ptrdiff_t offset =
        (SafeInt<ptrdiff_t>(batch) * parameters.q_sequence_length * parameters.q_num_heads +
         SafeInt<ptrdiff_t>(q_index) * parameters.q_num_heads + q_head) *
        parameters.v_head_size;
    return output + offset;
  }

  const ptrdiff_t offset =
      ((SafeInt<ptrdiff_t>(batch) * parameters.q_num_heads + q_head) *
           parameters.q_sequence_length +
       q_index) *
      parameters.v_head_size;
  return output + offset;
}

float DotProduct(const float* lhs, const float* rhs, int count) {
  float result = 0.0f;
  for (int i = 0; i < count; ++i) {
    result += lhs[i] * rhs[i];
  }
  return result;
}

float Softcap(float score, float softcap) {
  return softcap * std::tanh(score / softcap);
}

float MaskValue(const Tensor* mask_index, const AttentionParameters& parameters,
                int batch, int q_head, int q_index, int kv_index) {
  if (mask_index == nullptr) {
    return 0.0f;
  }

  const int mask_dims = static_cast<int>(mask_index->Shape().NumDimensions());
  const int mask_batch_size = mask_dims < 4 ? 1 : static_cast<int>(mask_index->Shape()[0]);
  const int mask_num_heads = mask_dims < 3 ? 1 : static_cast<int>(mask_index->Shape()[mask_dims - 3]);
  const int mask_batch = batch % mask_batch_size;
  const int mask_head = q_head % mask_num_heads;

  ptrdiff_t offset = q_index * parameters.total_sequence_length + kv_index;
  if (mask_dims == 3) {
    offset += SafeInt<ptrdiff_t>(mask_head) * parameters.q_sequence_length * parameters.total_sequence_length;
  } else if (mask_dims == 4) {
    offset += (SafeInt<ptrdiff_t>(mask_batch) * mask_num_heads + mask_head) *
              parameters.q_sequence_length * parameters.total_sequence_length;
  }

  if (mask_index->IsDataType<bool>()) {
    return mask_index->Data<bool>()[offset] ? 0.0f : mask_filter_value<float>();
  }

  return mask_index->Data<float>()[offset];
}

bool IsMaskedByCausal(const AttentionParameters& parameters, int q_index, int kv_index) {
  return parameters.is_causal &&
         parameters.q_sequence_length > 1 &&
         kv_index > parameters.past_sequence_length + q_index;
}

bool IsMaskedByNonPad(const AttentionParameters& parameters, int batch, int kv_index) {
  return parameters.has_nonpad_kv_seqlen &&
         kv_index >= static_cast<int>(parameters.nonpad_kv_seqlen_data[batch]);
}

}  // namespace

bool CanUseOnnxFlashAttention(const AttentionParameters& parameters,
                              const Tensor* mask_index,
                              const Tensor* past_key,
                              const Tensor* past_value,
                              const Tensor* present_key,
                              const Tensor* present_value,
                              const Tensor* output_qk) {
  // Supported envelope for the CPU row-streaming prototype:
  //   - dtype float only (enforced by the caller before this helper)
  //   - 3D packed/strided and 4D BNSH layouts
  //   - no cache/present outputs and no qk_matmul_output snapshots
  //   - fp32 softmax, causal, bool/numeric masks, softcap, nonpad, and GQA
  // TODO(cpu-flex-attention): move this into the mapped MLAS tiled MlasOnnxFlashAttention
  // API before enabling it from auto selection or claiming FlashAttention-style coverage.
  if (parameters.softmax_precision != 0 ||
      parameters.qk_matmul_output_mode != attention_helper::QKMatMulOutputMode::kNone ||
      past_key != nullptr ||
      past_value != nullptr ||
      present_key != nullptr ||
      present_value != nullptr ||
      output_qk != nullptr) {
    return false;
  }

  if (parameters.has_nonpad_kv_seqlen) {
    for (int batch = 0; batch < parameters.batch_size; ++batch) {
      // All-masked rows currently rely on materialized CPU softmax's finite-sentinel behavior.
      if (parameters.nonpad_kv_seqlen_data[batch] == 0) {
        return false;
      }
    }
  }

  return mask_index == nullptr || mask_index->IsDataType<bool>() || mask_index->IsDataType<float>();
}

Status DispatchOnnxFlashAttention(const float* Q,
                                  const float* K,
                                  const float* V,
                                  const Tensor* mask_index,
                                  float* output,
                                  const AttentionParameters& parameters,
                                  ThreadPool* tp) {
  TensorOpCost unit_cost;
  unit_cost.compute_cycles =
      static_cast<double>(SafeInt<ptrdiff_t>(2) * parameters.head_size *
                          parameters.total_sequence_length * parameters.v_head_size);
  unit_cost.bytes_loaded =
      static_cast<double>(SafeInt<ptrdiff_t>(parameters.head_size + parameters.v_head_size) *
                          parameters.total_sequence_length * sizeof(float));
  unit_cost.bytes_stored = static_cast<double>(parameters.v_head_size * sizeof(float));

  const ptrdiff_t task_count = SafeInt<ptrdiff_t>(parameters.batch_size) *
                               parameters.q_num_heads *
                               parameters.q_sequence_length;
  ThreadPool::TryParallelFor(tp, task_count, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    std::vector<float> scores(parameters.total_sequence_length);
    std::vector<float> accumulator(parameters.v_head_size);

    for (std::ptrdiff_t task_index = begin; task_index != end; ++task_index) {
      int q_index = static_cast<int>(task_index % parameters.q_sequence_length);
      std::ptrdiff_t batch_head = task_index / parameters.q_sequence_length;
      int q_head = static_cast<int>(batch_head % parameters.q_num_heads);
      int batch = static_cast<int>(batch_head / parameters.q_num_heads);
      int kv_head = q_head * parameters.kv_num_heads / parameters.q_num_heads;

      const float* q = QueryPointer(Q, parameters, batch, q_head, q_index);
      float row_max = std::numeric_limits<float>::lowest();
      for (int kv_index = 0; kv_index < parameters.total_sequence_length; ++kv_index) {
        float score;
        if (IsMaskedByCausal(parameters, q_index, kv_index) ||
            IsMaskedByNonPad(parameters, batch, kv_index)) {
          score = mask_filter_value<float>();
        } else {
          const float* k = KeyPointer(K, parameters, batch, kv_head, kv_index);
          score = parameters.scale * DotProduct(q, k, parameters.head_size);
          if (parameters.softcap > 0.0f) {
            score = Softcap(score, parameters.softcap);
          }
          score += MaskValue(mask_index, parameters, batch, q_head, q_index, kv_index);
        }

        scores[kv_index] = score;
        row_max = std::max(row_max, score);
      }

      float row_sum = 0.0f;
      for (int kv_index = 0; kv_index < parameters.total_sequence_length; ++kv_index) {
        scores[kv_index] = std::exp(scores[kv_index] - row_max);
        row_sum += scores[kv_index];
      }

      std::fill(accumulator.begin(), accumulator.end(), 0.0f);
      for (int kv_index = 0; kv_index < parameters.total_sequence_length; ++kv_index) {
        const float probability = scores[kv_index] / row_sum;
        const float* v = ValuePointer(V, parameters, batch, kv_head, kv_index);
        for (int v_index = 0; v_index < parameters.v_head_size; ++v_index) {
          accumulator[v_index] += probability * v[v_index];
        }
      }

      float* y = OutputPointer(output, parameters, batch, q_head, q_index);
      std::copy(accumulator.begin(), accumulator.end(), y);
    }
  });

  return Status::OK();
}

}  // namespace onnxruntime
