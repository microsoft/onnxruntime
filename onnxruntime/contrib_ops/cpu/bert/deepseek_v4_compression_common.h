// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

namespace deepseek_v4_attention_impl {

enum class DeepSeekV4AttentionMode {
  kSliding,
  kCsa,
  kHca,
};

constexpr int kFirstHcaInputIndex = 17;
constexpr int kLastHcaInputIndex = 23;
constexpr int kFirstCsaInputIndex = 24;
constexpr int kLastCsaInputIndex = 43;

inline bool HasInput(const OpKernelContext* context, int index) {
  return index < context->InputCount() && context->Input<Tensor>(index) != nullptr;
}

inline bool HasAnyInputInRange(const OpKernelContext* context, int begin, int end) {
  for (int i = begin; i <= end; ++i) {
    if (HasInput(context, i)) {
      return true;
    }
  }
  return false;
}

inline DeepSeekV4AttentionMode ParseAttentionMode(const std::string& mode) {
  if (mode == "sliding") {
    return DeepSeekV4AttentionMode::kSliding;
  }
  if (mode == "csa") {
    return DeepSeekV4AttentionMode::kCsa;
  }
  if (mode == "hca") {
    return DeepSeekV4AttentionMode::kHca;
  }
  ORT_THROW("DeepSeekV4Attention: unsupported attention_mode='", mode, "'.");
}

template <typename T>
float ToFloat(T v) {
  return static_cast<float>(v);
}

template <typename T>
T FromFloat(float v) {
  if constexpr (std::is_same_v<T, float>) {
    return v;
  } else {
    return T(v);
  }
}

inline void MatMulRowMajor(const float* a, int64_t m, int64_t k, const float* b, int64_t n, std::vector<float>& out) {
  out.assign(static_cast<size_t>(m * n), 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int64_t kk = 0; kk < k; ++kk) {
        sum += a[row * k + kk] * b[kk * n + col];
      }
      out[row * n + col] = sum;
    }
  }
}

inline void ApplyRmsNorm(std::vector<float>& values, const float* weight, int64_t dim, float epsilon) {
  float squares_sum = 0.0f;
  for (int64_t i = 0; i < dim; ++i) {
    squares_sum += values[static_cast<size_t>(i)] * values[static_cast<size_t>(i)];
  }
  const float inv_rms = 1.0f / std::sqrt(squares_sum / static_cast<float>(dim) + epsilon);
  for (int64_t i = 0; i < dim; ++i) {
    values[static_cast<size_t>(i)] *= inv_rms * weight[i];
  }
}

template <typename T>
void CopyTensor(const Tensor& src, Tensor& dst) {
  ORT_ENFORCE(src.Shape() == dst.Shape(), "CopyTensor shape mismatch.");
  const auto* src_data = src.Data<T>();
  auto* dst_data = dst.MutableData<T>();
  const size_t count = src.Shape().Size();
  if (count > 0) {
    std::memcpy(dst_data, src_data, count * sizeof(T));
  }
}

template <typename T>
float ReadAttentionBias(const Tensor* attention_bias,
                        int64_t batch,
                        int64_t head,
                        int64_t query,
                        int64_t key_total_index) {
  if (attention_bias == nullptr) {
    return 0.0f;
  }

  const auto& shape = attention_bias->Shape();
  ORT_ENFORCE(shape.NumDimensions() == 4, "attention_bias must have rank 4.");
  const int64_t b_dim = shape[0];
  const int64_t h_dim = shape[1];
  const int64_t q_dim = shape[2];
  const int64_t k_dim = shape[3];

  ORT_ENFORCE((b_dim == 1 || b_dim > batch) && (h_dim == 1 || h_dim > head) && q_dim > query,
              "attention_bias shape is incompatible with current index.");
  if (k_dim <= key_total_index) {
    return 0.0f;
  }

  const int64_t b_idx = b_dim == 1 ? 0 : batch;
  const int64_t h_idx = h_dim == 1 ? 0 : head;
  const int64_t index = ((b_idx * h_dim + h_idx) * q_dim + query) * k_dim + key_total_index;
  return ToFloat(attention_bias->Data<T>()[index]);
}

inline void ApplyInterleavedTrailingRope(std::vector<float>& head_values,
                                  int64_t head_size,
                                  int64_t rotary_dim,
                                  const float* cos_row,
                                  const float* sin_row) {
  const int64_t start = head_size - rotary_dim;
  for (int64_t i = 0; i < rotary_dim; i += 2) {
    const float x0 = head_values[static_cast<size_t>(start + i)];
    const float x1 = head_values[static_cast<size_t>(start + i + 1)];
    const float c = cos_row[i / 2];
    const float s = sin_row[i / 2];
    head_values[static_cast<size_t>(start + i)] = x0 * c - x1 * s;
    head_values[static_cast<size_t>(start + i + 1)] = x0 * s + x1 * c;
  }
}

template <typename T>
std::vector<float> ToFloatVector(const Tensor& tensor) {
  const T* data = tensor.Data<T>();
  std::vector<float> result(static_cast<size_t>(tensor.Shape().Size()), 0.0f);
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = ToFloat(data[i]);
  }
  return result;
}

template <typename T>
void WriteFloatVector(Tensor& tensor, const std::vector<float>& values) {
  ORT_ENFORCE(tensor.Shape().Size() == static_cast<int64_t>(values.size()), "WriteFloatVector shape mismatch.");
  T* data = tensor.MutableData<T>();
  for (size_t i = 0; i < values.size(); ++i) {
    data[i] = FromFloat<T>(values[i]);
  }
}

inline std::vector<float> MakeRows(const std::vector<float>& a, int64_t rows, int64_t k,
                            const std::vector<float>& b, int64_t n) {
  std::vector<float> out;
  MatMulRowMajor(a.data(), rows, k, b.data(), n, out);
  return out;
}

inline void ApplyRmsNormRows(std::vector<float>& values, const float* weight, int64_t rows, int64_t dim, float epsilon) {
  for (int64_t r = 0; r < rows; ++r) {
    float squares_sum = 0.0f;
    float* row = values.data() + r * dim;
    for (int64_t d = 0; d < dim; ++d) {
      squares_sum += row[d] * row[d];
    }
    const float inv_rms = 1.0f / std::sqrt(squares_sum / static_cast<float>(dim) + epsilon);
    for (int64_t d = 0; d < dim; ++d) {
      row[d] *= inv_rms * weight[d];
    }
  }
}

struct EntryState {
  std::vector<float> data;
  int64_t entries{};
  bool rank4{};
};

inline EntryState ReadEntries(const Tensor& tensor, int64_t batch_size, int64_t head_size) {
  const auto& shape = tensor.Shape();
  ORT_ENFORCE(shape.NumDimensions() == 3 || shape.NumDimensions() == 4, "entries state must be rank 3 or 4.");
  EntryState state;
  state.rank4 = shape.NumDimensions() == 4;
  if (state.rank4) {
    ORT_ENFORCE(shape[0] == batch_size && shape[1] == 1 && shape[3] == head_size,
                "rank-4 entries state must have shape (B, 1, E, head_size).");
    state.entries = shape[2];
  } else {
    ORT_ENFORCE(shape[0] == batch_size && shape[2] == head_size,
                "rank-3 entries state must have shape (B, E, head_size).");
    state.entries = shape[1];
  }
  return state;
}

inline TensorShape EntryOutputShape(int64_t batch_size, int64_t entries, int64_t head_size, bool rank4) {
  return rank4 ? TensorShape({batch_size, 1, entries, head_size})
               : TensorShape({batch_size, entries, head_size});
}

inline std::vector<float> ReadEntryData(const std::vector<float>& flat, const EntryState& state,
                                 int64_t batch_size, int64_t head_size) {
  std::vector<float> result(static_cast<size_t>(batch_size * state.entries * head_size), 0.0f);
  if (!state.rank4) {
    std::copy(flat.begin(), flat.end(), result.begin());
    return result;
  }
  for (int64_t b = 0; b < batch_size; ++b) {
    const size_t src = static_cast<size_t>(((b * 1) * state.entries) * head_size);
    const size_t dst = static_cast<size_t>(b * state.entries * head_size);
    std::copy(flat.begin() + src, flat.begin() + src + static_cast<size_t>(state.entries * head_size),
              result.begin() + dst);
  }
  return result;
}

inline std::vector<float> WriteEntryData(const std::vector<float>& internal, int64_t batch_size,
                                  int64_t entries, int64_t head_size, bool rank4) {
  if (!rank4) {
    return internal;
  }
  std::vector<float> result(static_cast<size_t>(batch_size * entries * head_size), 0.0f);
  for (int64_t b = 0; b < batch_size; ++b) {
    const size_t src = static_cast<size_t>(b * entries * head_size);
    const size_t dst = static_cast<size_t>(((b * 1) * entries) * head_size);
    std::copy(internal.begin() + src, internal.begin() + src + static_cast<size_t>(entries * head_size),
              result.begin() + dst);
  }
  return result;
}

inline void AppendEntries(std::vector<float>& entries, int64_t old_entries, const std::vector<float>& new_entries,
                   int64_t new_count, int64_t batch_size, int64_t head_size) {
  if (new_count == 0) {
    return;
  }
  std::vector<float> combined(static_cast<size_t>(batch_size * (old_entries + new_count) * head_size), 0.0f);
  for (int64_t b = 0; b < batch_size; ++b) {
    const size_t old_src = static_cast<size_t>(b * old_entries * head_size);
    const size_t dst = static_cast<size_t>(b * (old_entries + new_count) * head_size);
    std::copy(entries.begin() + old_src, entries.begin() + old_src + static_cast<size_t>(old_entries * head_size),
              combined.begin() + dst);
    const size_t new_src = static_cast<size_t>(b * new_count * head_size);
    std::copy(new_entries.begin() + new_src, new_entries.begin() + new_src + static_cast<size_t>(new_count * head_size),
              combined.begin() + dst + static_cast<size_t>(old_entries * head_size));
  }
  entries.swap(combined);
}

inline std::vector<float> CompressWindows(const std::vector<float>& window_kv,
                                   const std::vector<float>& window_gate,
                                   int64_t batch_size,
                                   int64_t num_windows,
                                   int64_t slots,
                                   int64_t head_size,
                                   const float* norm_weight,
                                   float epsilon) {
  std::vector<float> compressed(static_cast<size_t>(batch_size * num_windows * head_size), 0.0f);
  std::vector<float> logits(static_cast<size_t>(slots), 0.0f);
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t w = 0; w < num_windows; ++w) {
      float* out = compressed.data() + (b * num_windows + w) * head_size;
      for (int64_t d = 0; d < head_size; ++d) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int64_t s = 0; s < slots; ++s) {
          const float logit = window_gate[((b * num_windows + w) * slots + s) * head_size + d];
          logits[static_cast<size_t>(s)] = logit;
          max_logit = std::max(max_logit, logit);
        }
        float sum = 0.0f;
        for (int64_t s = 0; s < slots; ++s) {
          const float e = std::exp(logits[static_cast<size_t>(s)] - max_logit);
          logits[static_cast<size_t>(s)] = e;
          sum += e;
        }
        if (sum == 0.0f) {
          continue;
        }
        for (int64_t s = 0; s < slots; ++s) {
          const float weight = logits[static_cast<size_t>(s)] / sum;
          out[d] += weight * window_kv[((b * num_windows + w) * slots + s) * head_size + d];
        }
      }
    }
  }
  ApplyRmsNormRows(compressed, norm_weight, batch_size * num_windows, head_size, epsilon);
  return compressed;
}

struct CompressorResult {
  std::vector<float> pending_kv;
  std::vector<float> pending_gate;
  std::vector<float> entries;
  std::vector<float> overlap_kv;
  std::vector<float> overlap_gate;
  int64_t pending_count{};
  int64_t entry_count{};
  bool entries_rank4{};
};

template <typename T>
Status RunCompressor(const Tensor& hidden,
                     const Tensor& cos_cache,
                     const Tensor& sin_cache,
                     const Tensor& kv_weight,
                     const Tensor& gate_weight,
                     const Tensor& position_bias,
                     const Tensor& norm_weight,
                     const Tensor& past_pending_kv,
                     const Tensor& past_pending_gate,
                     const Tensor& past_entries,
                     const Tensor* past_overlap_kv,
                     const Tensor* past_overlap_gate,
                     int64_t compress_rate,
                     int64_t rotary_dim,
                     float epsilon,
                     CompressorResult& result,
                     int64_t entry_capacity = 0,
                     const Tensor* position_ids = nullptr) {
  const bool fixed_mode = entry_capacity > 0;
  const auto& hidden_shape = hidden.Shape();
  ORT_RETURN_IF_NOT(hidden_shape.NumDimensions() == 3, "hidden_states must have rank 3.");
  const int64_t batch = hidden_shape[0];
  const int64_t sequence = hidden_shape[1];
  const int64_t hidden_size = hidden_shape[2];
  ORT_RETURN_IF_NOT(cos_cache.Shape() == sin_cache.Shape() && cos_cache.Shape().NumDimensions() == 2,
                    "cos_cache and sin_cache must have the same rank-2 shape.");
  const bool is_overlap = past_overlap_kv != nullptr || past_overlap_gate != nullptr;
  ORT_RETURN_IF_NOT((past_overlap_kv == nullptr) == (past_overlap_gate == nullptr),
                    "overlap KV and gate states must either both be provided or both be omitted.");
  ORT_RETURN_IF_NOT(kv_weight.Shape().NumDimensions() == 2 && kv_weight.Shape()[0] == hidden_size,
                    "kv_weight must have shape (D, H) or (D, 2 * H).");
  const int64_t width = kv_weight.Shape()[1];
  ORT_RETURN_IF(is_overlap && width % 2 != 0, "overlap compressor width must be even.");
  const int64_t head_size = is_overlap ? width / 2 : width;
  ORT_RETURN_IF_NOT(gate_weight.Shape() == kv_weight.Shape(),
                    "gate_weight must have the same shape as kv_weight.");
  ORT_RETURN_IF_NOT(position_bias.Shape() == TensorShape({compress_rate, width}),
                    "position_bias must have shape (compress_rate, projection width).");
  ORT_RETURN_IF_NOT(norm_weight.Shape() == TensorShape({head_size}),
                    "norm_weight must have shape (H).");
  ORT_RETURN_IF_NOT(rotary_dim <= head_size && cos_cache.Shape()[1] * 2 >= rotary_dim,
                    "rotary dimensions are incompatible with H and the caches.");
  ORT_RETURN_IF_NOT(past_pending_kv.Shape().NumDimensions() == 3 && past_pending_kv.Shape()[0] == batch &&
                        past_pending_kv.Shape()[2] == width && past_pending_gate.Shape() == past_pending_kv.Shape(),
            "pending states must have shape (B, P, projection width).");
  if (fixed_mode) {
    ORT_RETURN_IF_NOT(past_pending_kv.Shape()[1] == compress_rate - 1,
                      "fixed-mode pending state must have capacity compress_rate - 1.");
  } else {
    ORT_RETURN_IF_NOT(past_pending_kv.Shape()[1] < compress_rate,
                      "pending state length must be less than compress_rate.");
  }
  if (is_overlap) {
    ORT_RETURN_IF_NOT(past_overlap_kv->Shape() == TensorShape({batch, compress_rate, head_size}) &&
                          past_overlap_gate->Shape() == past_overlap_kv->Shape(),
                      "overlap states must have shape (B, compress_rate, H).");
  }

  EntryState entry_state = ReadEntries(past_entries, batch, head_size);
  // In fixed mode derive logical counts from position_ids[b,0].
  int64_t pending_count;
  int64_t old_entry_count_derived;
  if (fixed_mode) {
    ORT_RETURN_IF_NOT(position_ids != nullptr, "position_ids required for fixed mode.");
    const int64_t start_pos = position_ids->Data<int64_t>()[0];
    pending_count = start_pos % compress_rate;
    old_entry_count_derived = start_pos / compress_rate;
  } else {
    pending_count = past_pending_kv.Shape()[1];
    old_entry_count_derived = entry_state.entries;
  }
  const int64_t total_count = pending_count + sequence;
  const int64_t usable_count = total_count / compress_rate * compress_rate;
  const int64_t new_entry_count = usable_count / compress_rate;
  result.pending_count = total_count - usable_count;
  result.entry_count = fixed_mode ? entry_capacity : (old_entry_count_derived + new_entry_count);
  result.entries_rank4 = entry_state.rank4;
  ORT_RETURN_IF(new_entry_count > 0 && (old_entry_count_derived + new_entry_count - 1) * compress_rate >= cos_cache.Shape()[0],
                "compressed entry position is outside the RoPE cache.");
  if (fixed_mode) {
    ORT_RETURN_IF(old_entry_count_derived + new_entry_count > entry_capacity,
                  "fixed entry_capacity exceeded.");
  }

  const auto hidden_data = ToFloatVector<T>(hidden);
  const auto kv_weight_data = ToFloatVector<T>(kv_weight);
  const auto gate_weight_data = ToFloatVector<T>(gate_weight);
  const auto bias_data = ToFloatVector<T>(position_bias);
  const auto norm_data = ToFloatVector<T>(norm_weight);
  const auto cos_data = ToFloatVector<T>(cos_cache);
  const auto sin_data = ToFloatVector<T>(sin_cache);
  auto current_kv = MakeRows(hidden_data, batch * sequence, hidden_size, kv_weight_data, width);
  auto current_gate = MakeRows(hidden_data, batch * sequence, hidden_size, gate_weight_data, width);

  auto combine = [&](const Tensor& pending, const std::vector<float>& current) {
    const auto pending_data = ToFloatVector<T>(pending);
    const int64_t pending_capacity = pending.Shape()[1];
    std::vector<float> combined(static_cast<size_t>(batch * total_count * width));
    for (int64_t b = 0; b < batch; ++b) {
      std::copy_n(pending_data.begin() + b * pending_capacity * width, pending_count * width,
                  combined.begin() + b * total_count * width);
      std::copy_n(current.begin() + b * sequence * width, sequence * width,
                  combined.begin() + (b * total_count + pending_count) * width);
    }
    return combined;
  };
  auto combined_kv = combine(past_pending_kv, current_kv);
  auto combined_gate = combine(past_pending_gate, current_gate);
  result.pending_kv.resize(static_cast<size_t>(batch * result.pending_count * width));
  result.pending_gate.resize(result.pending_kv.size());
  for (int64_t b = 0; b < batch; ++b) {
    std::copy_n(combined_kv.begin() + (b * total_count + usable_count) * width,
                result.pending_count * width, result.pending_kv.begin() + b * result.pending_count * width);
    std::copy_n(combined_gate.begin() + (b * total_count + usable_count) * width,
                result.pending_count * width, result.pending_gate.begin() + b * result.pending_count * width);
  }

  if (is_overlap) {
    result.overlap_kv = ToFloatVector<T>(*past_overlap_kv);
    result.overlap_gate = ToFloatVector<T>(*past_overlap_gate);
  }
  std::vector<float> new_entries;
  if (new_entry_count > 0) {
    std::vector<float> chunks_kv(static_cast<size_t>(batch * usable_count * width));
    std::vector<float> chunks_gate(chunks_kv.size());
    for (int64_t b = 0; b < batch; ++b) {
      std::copy_n(combined_kv.begin() + b * total_count * width, usable_count * width,
                  chunks_kv.begin() + b * usable_count * width);
      std::copy_n(combined_gate.begin() + b * total_count * width, usable_count * width,
                  chunks_gate.begin() + b * usable_count * width);
    }
    for (int64_t row = 0; row < batch * usable_count; ++row) {
      for (int64_t d = 0; d < width; ++d) {
        chunks_gate[static_cast<size_t>(row * width + d)] +=
            bias_data[static_cast<size_t>((row % compress_rate) * width + d)];
      }
    }

    const int64_t slots = is_overlap ? 2 * compress_rate : compress_rate;
    std::vector<float> window_kv(static_cast<size_t>(batch * new_entry_count * slots * head_size));
    std::vector<float> window_gate(window_kv.size());
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t window = 0; window < new_entry_count; ++window) {
        for (int64_t slot = 0; slot < compress_rate; ++slot) {
          for (int64_t d = 0; d < head_size; ++d) {
            const size_t current = static_cast<size_t>(((b * new_entry_count + window) * compress_rate + slot) * width + d);
            const size_t destination = static_cast<size_t>(((b * new_entry_count + window) * slots + slot) * head_size + d);
            if (!is_overlap) {
              window_kv[destination] = chunks_kv[current];
              window_gate[destination] = chunks_gate[current];
              continue;
            }
            const size_t cb_destination = destination + static_cast<size_t>(compress_rate * head_size);
            const size_t current_cb = current + static_cast<size_t>(head_size);
            const size_t previous = window == 0
                                        ? static_cast<size_t>((b * compress_rate + slot) * head_size + d)
                                        : static_cast<size_t>(((b * new_entry_count + window - 1) * compress_rate + slot) * width + d);
            window_kv[destination] = window == 0 ? result.overlap_kv[previous] : chunks_kv[previous];
            window_gate[destination] = window == 0 ? result.overlap_gate[previous] : chunks_gate[previous];
            window_kv[cb_destination] = chunks_kv[current_cb];
            window_gate[cb_destination] = chunks_gate[current_cb];
          }
        }
      }
    }
    new_entries = CompressWindows(window_kv, window_gate, batch, new_entry_count, slots, head_size,
                                  norm_data.data(), epsilon);
    const int64_t cache_width = cos_cache.Shape()[1];
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t entry = 0; entry < new_entry_count; ++entry) {
        const int64_t position = (old_entry_count_derived + entry) * compress_rate;
        const size_t offset = static_cast<size_t>((b * new_entry_count + entry) * head_size);
        std::vector<float> row(new_entries.begin() + offset, new_entries.begin() + offset + head_size);
        ApplyInterleavedTrailingRope(row, head_size, rotary_dim, cos_data.data() + position * cache_width,
                                     sin_data.data() + position * cache_width);
        std::copy(row.begin(), row.end(), new_entries.begin() + offset);
      }
      if (is_overlap) {
        for (int64_t slot = 0; slot < compress_rate; ++slot) {
          const size_t source = static_cast<size_t>(((b * new_entry_count + new_entry_count - 1) * compress_rate + slot) * width);
          const size_t destination = static_cast<size_t>((b * compress_rate + slot) * head_size);
          std::copy_n(chunks_kv.begin() + source, head_size, result.overlap_kv.begin() + destination);
          std::copy_n(chunks_gate.begin() + source, head_size, result.overlap_gate.begin() + destination);
        }
      }
    }
  }

  if (fixed_mode) {
    // Fixed mode: result.entries has capacity layout [batch * entry_capacity * head_size].
    result.entries.assign(static_cast<size_t>(batch * entry_capacity * head_size), 0.0f);
    const auto old_flat = ToFloatVector<T>(past_entries);
    for (int64_t b = 0; b < batch; ++b) {
      const size_t src_start = static_cast<size_t>(b * entry_state.entries * head_size);
      const size_t dst_start = static_cast<size_t>(b * entry_capacity * head_size);
      std::copy_n(old_flat.begin() + src_start, static_cast<size_t>(old_entry_count_derived * head_size),
                  result.entries.begin() + dst_start);
      for (int64_t e = 0; e < new_entry_count; ++e) {
        const size_t new_src = static_cast<size_t>((b * new_entry_count + e) * head_size);
        const size_t new_dst = dst_start + static_cast<size_t>((old_entry_count_derived + e) * head_size);
        std::copy_n(new_entries.begin() + new_src, static_cast<size_t>(head_size),
                    result.entries.begin() + new_dst);
      }
    }
  } else {
    result.entries = ReadEntryData(ToFloatVector<T>(past_entries), entry_state, batch, head_size);
    AppendEntries(result.entries, entry_state.entries, new_entries, new_entry_count, batch, head_size);
  }
  return Status::OK();
}

}  // namespace deepseek_v4_attention_impl
}  // namespace contrib
}  // namespace onnxruntime
