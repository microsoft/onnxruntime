// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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

DeepSeekV4AttentionMode ParseAttentionMode(const std::string& mode) {
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

void MatMulRowMajor(const float* a, int64_t m, int64_t k, const float* b, int64_t n, std::vector<float>& out) {
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

void ApplyRmsNorm(std::vector<float>& values, const float* weight, int64_t dim, float epsilon) {
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

void ApplyInterleavedTrailingRope(std::vector<float>& head_values,
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

std::vector<float> MakeRows(const std::vector<float>& a, int64_t rows, int64_t k,
                            const std::vector<float>& b, int64_t n) {
  std::vector<float> out;
  MatMulRowMajor(a.data(), rows, k, b.data(), n, out);
  return out;
}

void ApplyRmsNormRows(std::vector<float>& values, const float* weight, int64_t rows, int64_t dim, float epsilon) {
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

EntryState ReadEntries(const Tensor& tensor, int64_t batch_size, int64_t head_size) {
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

TensorShape EntryOutputShape(int64_t batch_size, int64_t entries, int64_t head_size, bool rank4) {
  return rank4 ? TensorShape({batch_size, 1, entries, head_size})
               : TensorShape({batch_size, entries, head_size});
}

std::vector<float> ReadEntryData(const std::vector<float>& flat, const EntryState& state,
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

std::vector<float> WriteEntryData(const std::vector<float>& internal, int64_t batch_size,
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

void AppendEntries(std::vector<float>& entries, int64_t old_entries, const std::vector<float>& new_entries,
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

std::vector<float> CompressWindows(const std::vector<float>& window_kv,
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

template <typename T>
class DeepSeekV4Attention final : public OpKernel {
 public:
  explicit DeepSeekV4Attention(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads_).IsOK() && num_heads_ > 0,
                "DeepSeekV4Attention: num_heads must be > 0.");

    ORT_ENFORCE(info.GetAttr("head_size", &head_size_).IsOK() && head_size_ > 0,
                "DeepSeekV4Attention: head_size must be > 0.");

    ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads_).IsOK() && kv_num_heads_ == 1,
                "DeepSeekV4Attention: kv_num_heads must be 1.");

    ORT_ENFORCE(info.GetAttr("q_lora_rank", &q_lora_rank_).IsOK() && q_lora_rank_ > 0,
                "DeepSeekV4Attention: q_lora_rank must be > 0.");

    ORT_ENFORCE(info.GetAttr("o_groups", &o_groups_).IsOK() && o_groups_ > 0,
                "DeepSeekV4Attention: o_groups must be > 0.");

    ORT_ENFORCE(info.GetAttr("o_lora_rank", &o_lora_rank_).IsOK() && o_lora_rank_ > 0,
                "DeepSeekV4Attention: o_lora_rank must be > 0.");

    ORT_ENFORCE(info.GetAttr("rotary_dim", &rotary_dim_).IsOK() && rotary_dim_ > 0 && rotary_dim_ <= head_size_,
                "DeepSeekV4Attention: rotary_dim must be in (0, head_size].");

    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1,
                "DeepSeekV4Attention: rotary_interleaved must be 1.");
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("rotary_trailing", 0) == 1,
                "DeepSeekV4Attention: rotary_trailing must be 1.");
    ORT_ENFORCE(info.GetAttrOrDefault<int64_t>("do_output_derotate", 0) == 1,
                "DeepSeekV4Attention: do_output_derotate must be 1.");

    ORT_ENFORCE(info.GetAttr("local_window_size", &local_window_size_).IsOK() && local_window_size_ > 0,
                "DeepSeekV4Attention: local_window_size must be > 0.");

    ORT_ENFORCE(info.GetAttr("rms_norm_epsilon", &rms_norm_epsilon_).IsOK() && rms_norm_epsilon_ > 0.0f,
                "DeepSeekV4Attention: rms_norm_epsilon must be > 0.");

    scale_ = info.GetAttrOrDefault<float>("scale", 1.0f / std::sqrt(static_cast<float>(head_size_)));

    const std::string attention_mode = info.GetAttrOrDefault<std::string>("attention_mode", "sliding");
    attention_mode_ = ParseAttentionMode(attention_mode);

    if (attention_mode_ == DeepSeekV4AttentionMode::kCsa ||
        attention_mode_ == DeepSeekV4AttentionMode::kHca) {
      float compress_rate = 0.0f;
      ORT_ENFORCE(info.GetAttr("compress_rate", &compress_rate).IsOK() && compress_rate > 0.0f,
                  "DeepSeekV4Attention: compress_rate must be provided and > 0 for csa/hca mode.");
      compress_rate_ = static_cast<int64_t>(compress_rate);
      ORT_ENFORCE(static_cast<float>(compress_rate_) == compress_rate,
                  "DeepSeekV4Attention: compress_rate must be an integer value.");
    }

    if (attention_mode_ == DeepSeekV4AttentionMode::kCsa) {
      ORT_ENFORCE(info.GetAttr("index_topk", &index_topk_).IsOK() && index_topk_ > 0,
                  "DeepSeekV4Attention: index_topk must be provided and > 0 for csa mode.");
      ORT_ENFORCE(info.GetAttr("index_num_heads", &index_num_heads_).IsOK() && index_num_heads_ > 0,
                  "DeepSeekV4Attention: index_num_heads must be provided and > 0 for csa mode.");
      ORT_ENFORCE(info.GetAttr("index_head_dim", &index_head_dim_).IsOK() && index_head_dim_ > 0,
                  "DeepSeekV4Attention: index_head_dim must be provided and > 0 for csa mode.");
    }
  }

  Status Compute(OpKernelContext* context) const override {
    const bool has_hca_inputs = HasAnyInputInRange(context, kFirstHcaInputIndex, kLastHcaInputIndex);
    const bool has_csa_inputs = HasAnyInputInRange(context, kFirstCsaInputIndex, kLastCsaInputIndex);

    if (attention_mode_ == DeepSeekV4AttentionMode::kSliding) {
      ORT_RETURN_IF(has_hca_inputs || has_csa_inputs,
                    "DeepSeekV4Attention(sliding): compressor inputs are not allowed.");
    } else if (attention_mode_ == DeepSeekV4AttentionMode::kHca) {
      ORT_RETURN_IF(!has_hca_inputs, "DeepSeekV4Attention(hca): HCA inputs are required.");
      ORT_RETURN_IF(has_csa_inputs, "DeepSeekV4Attention(hca): CSA inputs are not allowed.");
    } else {
      ORT_RETURN_IF(!has_csa_inputs, "DeepSeekV4Attention(csa): CSA inputs are required.");
      ORT_RETURN_IF(has_hca_inputs, "DeepSeekV4Attention(csa): HCA inputs are not allowed.");
    }

    const Tensor* hidden_states = context->Input<Tensor>(0);
    const Tensor* position_ids = context->Input<Tensor>(1);
    const Tensor* attention_bias = context->Input<Tensor>(2);
    const Tensor* past_key = context->Input<Tensor>(3);
    const Tensor* past_value = context->Input<Tensor>(4);
    const Tensor* seqlens_k = context->Input<Tensor>(5);
    const Tensor* total_sequence_length = context->Input<Tensor>(6);
    const Tensor* cos_cache = context->Input<Tensor>(7);
    const Tensor* sin_cache = context->Input<Tensor>(8);
    const Tensor* q_a_weight = context->Input<Tensor>(9);
    const Tensor* q_a_norm_weight = context->Input<Tensor>(10);
    const Tensor* q_b_weight = context->Input<Tensor>(11);
    const Tensor* kv_weight = context->Input<Tensor>(12);
    const Tensor* kv_norm_weight = context->Input<Tensor>(13);
    const Tensor* o_a_weight = context->Input<Tensor>(14);
    const Tensor* o_b_weight = context->Input<Tensor>(15);
    const Tensor* head_sink = context->Input<Tensor>(16);
    const Tensor* hca_kv_weight = context->Input<Tensor>(17);
    const Tensor* hca_gate_weight = context->Input<Tensor>(18);
    const Tensor* hca_position_bias = context->Input<Tensor>(19);
    const Tensor* hca_kv_norm_weight = context->Input<Tensor>(20);
    const Tensor* past_hca_pending_kv = context->Input<Tensor>(21);
    const Tensor* past_hca_pending_gate = context->Input<Tensor>(22);
    const Tensor* past_hca_entries = context->Input<Tensor>(23);
    const Tensor* csa_kv_weight = context->Input<Tensor>(24);
    const Tensor* csa_gate_weight = context->Input<Tensor>(25);
    const Tensor* csa_position_bias = context->Input<Tensor>(26);
    const Tensor* csa_kv_norm_weight = context->Input<Tensor>(27);
    const Tensor* index_kv_weight = context->Input<Tensor>(28);
    const Tensor* index_gate_weight = context->Input<Tensor>(29);
    const Tensor* index_position_bias = context->Input<Tensor>(30);
    const Tensor* index_kv_norm_weight = context->Input<Tensor>(31);
    const Tensor* index_q_b_weight = context->Input<Tensor>(32);
    const Tensor* index_weights_proj_weight = context->Input<Tensor>(33);
    const Tensor* past_csa_pending_kv = context->Input<Tensor>(34);
    const Tensor* past_csa_pending_gate = context->Input<Tensor>(35);
    const Tensor* past_csa_entries = context->Input<Tensor>(36);
    const Tensor* past_csa_overlap_kv = context->Input<Tensor>(37);
    const Tensor* past_csa_overlap_gate = context->Input<Tensor>(38);
    const Tensor* past_index_pending_kv = context->Input<Tensor>(39);
    const Tensor* past_index_pending_gate = context->Input<Tensor>(40);
    const Tensor* past_index_entries = context->Input<Tensor>(41);
    const Tensor* past_index_overlap_kv = context->Input<Tensor>(42);
    const Tensor* past_index_overlap_gate = context->Input<Tensor>(43);

    ORT_RETURN_IF_NOT(hidden_states != nullptr && position_ids != nullptr &&
                          past_key != nullptr && past_value != nullptr &&
                          seqlens_k != nullptr && total_sequence_length != nullptr &&
                          cos_cache != nullptr && sin_cache != nullptr &&
                          q_a_weight != nullptr && q_a_norm_weight != nullptr &&
                          q_b_weight != nullptr && kv_weight != nullptr &&
                          kv_norm_weight != nullptr && o_a_weight != nullptr &&
                          o_b_weight != nullptr && head_sink != nullptr,
                      "DeepSeekV4Attention: required input is missing.");
    if (attention_mode_ == DeepSeekV4AttentionMode::kHca) {
      ORT_RETURN_IF_NOT(hca_kv_weight && hca_gate_weight && hca_position_bias && hca_kv_norm_weight &&
                            past_hca_pending_kv && past_hca_pending_gate && past_hca_entries,
                        "DeepSeekV4Attention(hca): all HCA inputs are required.");
    }
    if (attention_mode_ == DeepSeekV4AttentionMode::kCsa) {
      ORT_RETURN_IF_NOT(csa_kv_weight && csa_gate_weight && csa_position_bias && csa_kv_norm_weight &&
                            index_kv_weight && index_gate_weight && index_position_bias && index_kv_norm_weight &&
                            index_q_b_weight && index_weights_proj_weight && past_csa_pending_kv &&
                            past_csa_pending_gate && past_csa_entries && past_csa_overlap_kv &&
                            past_csa_overlap_gate && past_index_pending_kv && past_index_pending_gate &&
                            past_index_entries && past_index_overlap_kv && past_index_overlap_gate,
                        "DeepSeekV4Attention(csa): all CSA inputs are required.");
    }

    const auto& hidden_shape = hidden_states->Shape();
    ORT_RETURN_IF_NOT(hidden_shape.NumDimensions() == 3, "hidden_states must be rank 3.");
    const int64_t batch_size = hidden_shape[0];
    const int64_t sequence_length = hidden_shape[1];
    const int64_t hidden_size = hidden_shape[2];
    ORT_RETURN_IF_NOT(hidden_size == num_heads_ * head_size_, "hidden size must match num_heads * head_size.");

    ORT_RETURN_IF_NOT(position_ids->Shape().NumDimensions() == 2 &&
                          position_ids->Shape()[0] == batch_size &&
                          position_ids->Shape()[1] == sequence_length,
                      "position_ids must have shape (B, S).");

    const auto& past_key_shape = past_key->Shape();
    const auto& past_value_shape = past_value->Shape();
    ORT_RETURN_IF_NOT(past_key_shape == past_value_shape, "past_key and past_value shapes must match.");
    ORT_RETURN_IF_NOT(past_key_shape.NumDimensions() == 4, "past_key/past_value must be rank 4.");
    ORT_RETURN_IF_NOT(past_key_shape[0] == batch_size &&
                          past_key_shape[1] == kv_num_heads_ &&
                          past_key_shape[3] == head_size_,
                      "past_key shape must be (B, kv_num_heads=1, C, head_size).");
    const int64_t cache_capacity = past_key_shape[2];

    ORT_RETURN_IF_NOT(seqlens_k->Shape().NumDimensions() == 1 && seqlens_k->Shape()[0] == batch_size,
                      "seqlens_k must have shape (B).");
    ORT_RETURN_IF_NOT(total_sequence_length->Shape().NumDimensions() == 1 && total_sequence_length->Shape()[0] == 1,
                      "total_sequence_length must have shape (1).");

    ORT_RETURN_IF_NOT(cos_cache->Shape().NumDimensions() == 2 && sin_cache->Shape().NumDimensions() == 2,
                      "cos_cache and sin_cache must be rank 2.");
    ORT_RETURN_IF_NOT(cos_cache->Shape() == sin_cache->Shape(), "cos_cache and sin_cache shapes must match.");
    ORT_RETURN_IF_NOT(cos_cache->Shape()[1] * 2 >= rotary_dim_, "cos/sin cache rotary width is too small.");

    ORT_RETURN_IF_NOT(q_a_weight->Shape().NumDimensions() == 2 &&
                          q_a_weight->Shape()[0] == hidden_size && q_a_weight->Shape()[1] == q_lora_rank_,
                      "q_a_weight shape mismatch.");
    ORT_RETURN_IF_NOT(q_a_norm_weight->Shape().NumDimensions() == 1 && q_a_norm_weight->Shape()[0] == q_lora_rank_,
                      "q_a_norm_weight shape mismatch.");
    ORT_RETURN_IF_NOT(q_b_weight->Shape().NumDimensions() == 2 &&
                          q_b_weight->Shape()[0] == q_lora_rank_ && q_b_weight->Shape()[1] == hidden_size,
                      "q_b_weight shape mismatch.");

    ORT_RETURN_IF_NOT(kv_weight->Shape().NumDimensions() == 2 && kv_weight->Shape()[0] == hidden_size,
                      "kv_weight shape mismatch.");
    const int64_t kv_width = kv_weight->Shape()[1];
    ORT_RETURN_IF_NOT(kv_width >= head_size_ * 2, "kv_weight width must be at least 2 * head_size.");

    ORT_RETURN_IF_NOT(kv_norm_weight->Shape().NumDimensions() == 1 && kv_norm_weight->Shape()[0] == kv_width,
                      "kv_norm_weight shape mismatch.");

    ORT_RETURN_IF_NOT(o_a_weight->Shape().NumDimensions() == 2 &&
                          o_a_weight->Shape()[0] == hidden_size &&
                          o_a_weight->Shape()[1] == o_groups_ * o_lora_rank_,
                      "o_a_weight shape mismatch.");
    ORT_RETURN_IF_NOT(o_b_weight->Shape().NumDimensions() == 2 &&
                          o_b_weight->Shape()[0] == o_groups_ * o_lora_rank_ &&
                          o_b_weight->Shape()[1] == hidden_size,
                      "o_b_weight shape mismatch.");

    ORT_RETURN_IF_NOT(head_sink->Shape().NumDimensions() <= 1,
                      "head_sink must be a scalar or 1D tensor.");
    const int64_t head_sink_count = head_sink->Shape().NumDimensions() == 0 ? 1 : head_sink->Shape()[0];
    ORT_RETURN_IF_NOT(head_sink_count == 1 || head_sink_count == num_heads_,
                      "head_sink must have 1 or num_heads entries.");

    Tensor* output = context->Output(0, hidden_shape);
    Tensor* present_key = context->Output(1, past_key_shape);
    Tensor* present_value = context->Output(2, past_value_shape);

    CopyTensor<T>(*past_key, *present_key);
    CopyTensor<T>(*past_value, *present_value);

    if (attention_mode_ == DeepSeekV4AttentionMode::kSliding) {
      const std::vector<std::pair<int, int>> passthrough = {
          {21, 3},  {22, 4},  {23, 5},  {34, 6},  {35, 7},  {36, 8},  {37, 9},
          {38, 10}, {39, 11}, {40, 12}, {41, 13}, {42, 14}, {43, 15}};

      for (const auto& [input_index, output_index] : passthrough) {
        if (output_index < context->OutputCount() && HasInput(context, input_index)) {
          const Tensor* input_tensor = context->Input<Tensor>(input_index);
          Tensor* output_tensor = context->Output(output_index, input_tensor->Shape());
          CopyTensor<T>(*input_tensor, *output_tensor);
        }
      }
    }

    if (16 < context->OutputCount()) {
      Tensor* output_qk = context->Output(16, hidden_shape);
      auto* qk_data = output_qk->MutableData<T>();
      for (int64_t i = 0; i < output_qk->Shape().Size(); ++i) {
        qk_data[i] = FromFloat<T>(0.0f);
      }
    }

    const auto hidden_data = ToFloatVector<T>(*hidden_states);
    const auto q_a_weight_data = ToFloatVector<T>(*q_a_weight);
    const auto q_a_norm_weight_data = ToFloatVector<T>(*q_a_norm_weight);
    const auto q_b_weight_data = ToFloatVector<T>(*q_b_weight);
    const auto kv_weight_data = ToFloatVector<T>(*kv_weight);
    const auto kv_norm_weight_data = ToFloatVector<T>(*kv_norm_weight);
    const auto o_a_weight_data = ToFloatVector<T>(*o_a_weight);
    const auto o_b_weight_data = ToFloatVector<T>(*o_b_weight);
    const auto head_sink_data = ToFloatVector<T>(*head_sink);
    const auto cos_data = ToFloatVector<T>(*cos_cache);
    const auto sin_data = ToFloatVector<T>(*sin_cache);

    const auto* position_data = position_ids->Data<int64_t>();
    const auto* seqlens_data = seqlens_k->Data<int32_t>();
    const int64_t cos_rows = cos_cache->Shape()[0];
    const int64_t cos_width = cos_cache->Shape()[1];

    std::vector<float> compressed_entries;
    int64_t compressed_entry_count = 0;
    std::vector<int64_t> csa_selected_indices;
    int64_t csa_selected_topk = 0;

    auto add_position_bias = [&](std::vector<float>& gate, const std::vector<float>& bias,
                                 int64_t rows, int64_t width) {
      ORT_ENFORCE(static_cast<int64_t>(bias.size()) == compress_rate_ * width,
                  "compressor position_bias shape mismatch.");
      for (int64_t r = 0; r < rows; ++r) {
        const int64_t pos = r % compress_rate_;
        for (int64_t d = 0; d < width; ++d) {
          gate[static_cast<size_t>(r * width + d)] += bias[static_cast<size_t>(pos * width + d)];
        }
      }
    };

    auto append_pending = [&](const Tensor& pending_tensor, const std::vector<float>& current,
                              int64_t width, int64_t& total_tokens) {
      const auto& pending_shape = pending_tensor.Shape();
      ORT_ENFORCE(pending_shape.NumDimensions() == 3 && pending_shape[0] == batch_size &&
                      pending_shape[2] == width,
                  "pending state must have shape (B, P, width).");
      const int64_t pending_tokens = pending_shape[1];
      total_tokens = pending_tokens + sequence_length;
      std::vector<float> combined(static_cast<size_t>(batch_size * total_tokens * width), 0.0f);
      const auto pending = ToFloatVector<T>(pending_tensor);
      for (int64_t b = 0; b < batch_size; ++b) {
        std::copy(pending.begin() + static_cast<size_t>(b * pending_tokens * width),
                  pending.begin() + static_cast<size_t>((b + 1) * pending_tokens * width),
                  combined.begin() + static_cast<size_t>(b * total_tokens * width));
        std::copy(current.begin() + static_cast<size_t>(b * sequence_length * width),
                  current.begin() + static_cast<size_t>((b + 1) * sequence_length * width),
                  combined.begin() + static_cast<size_t>((b * total_tokens + pending_tokens) * width));
      }
      return combined;
    };

    auto write_pending_output = [&](int output_index, const std::vector<float>& combined,
                                    int64_t total_tokens, int64_t usable_tokens, int64_t width) {
      if (output_index >= context->OutputCount()) {
        return;
      }
      const int64_t pending_tokens = total_tokens - usable_tokens;
      std::vector<float> pending(static_cast<size_t>(batch_size * pending_tokens * width), 0.0f);
      for (int64_t b = 0; b < batch_size; ++b) {
        std::copy(combined.begin() + static_cast<size_t>((b * total_tokens + usable_tokens) * width),
                  combined.begin() + static_cast<size_t>((b * total_tokens + total_tokens) * width),
                  pending.begin() + static_cast<size_t>(b * pending_tokens * width));
      }
      Tensor* output_tensor = context->Output(output_index, TensorShape({batch_size, pending_tokens, width}));
      WriteFloatVector<T>(*output_tensor, pending);
    };

    auto make_hca_compressed = [&]() -> Status {
      ORT_RETURN_IF_NOT(hca_kv_weight->Shape().NumDimensions() == 2 && hca_kv_weight->Shape()[0] == hidden_size &&
                            hca_kv_weight->Shape()[1] == head_size_,
                        "hca_kv_weight shape mismatch.");
      ORT_RETURN_IF_NOT(hca_gate_weight->Shape() == hca_kv_weight->Shape(), "hca_gate_weight shape mismatch.");
      ORT_RETURN_IF_NOT(hca_position_bias->Shape().NumDimensions() == 2 &&
                            hca_position_bias->Shape()[0] == compress_rate_ &&
                            hca_position_bias->Shape()[1] == head_size_,
                        "hca_position_bias shape mismatch.");
      ORT_RETURN_IF_NOT(hca_kv_norm_weight->Shape().NumDimensions() == 1 &&
                            hca_kv_norm_weight->Shape()[0] == head_size_,
                        "hca_kv_norm_weight shape mismatch.");

      const auto hca_kv_w = ToFloatVector<T>(*hca_kv_weight);
      const auto hca_gate_w = ToFloatVector<T>(*hca_gate_weight);
      const auto hca_pos = ToFloatVector<T>(*hca_position_bias);
      const auto hca_norm = ToFloatVector<T>(*hca_kv_norm_weight);
      auto kv = MakeRows(hidden_data, batch_size * sequence_length, hidden_size, hca_kv_w, head_size_);
      auto gate = MakeRows(hidden_data, batch_size * sequence_length, hidden_size, hca_gate_w, head_size_);

      int64_t total_tokens = 0;
      auto combined_kv = append_pending(*past_hca_pending_kv, kv, head_size_, total_tokens);
      auto combined_gate = append_pending(*past_hca_pending_gate, gate, head_size_, total_tokens);
      const int64_t usable_tokens = (total_tokens / compress_rate_) * compress_rate_;
      const int64_t new_count = usable_tokens / compress_rate_;
      write_pending_output(3, combined_kv, total_tokens, usable_tokens, head_size_);
      write_pending_output(4, combined_gate, total_tokens, usable_tokens, head_size_);

      std::vector<float> new_entries;
      if (new_count > 0) {
        std::vector<float> window_kv(static_cast<size_t>(batch_size * new_count * compress_rate_ * head_size_), 0.0f);
        std::vector<float> window_gate(window_kv.size(), 0.0f);
        for (int64_t b = 0; b < batch_size; ++b) {
          std::copy(combined_kv.begin() + static_cast<size_t>(b * total_tokens * head_size_),
                    combined_kv.begin() + static_cast<size_t>((b * total_tokens + usable_tokens) * head_size_),
                    window_kv.begin() + static_cast<size_t>(b * usable_tokens * head_size_));
          std::copy(combined_gate.begin() + static_cast<size_t>(b * total_tokens * head_size_),
                    combined_gate.begin() + static_cast<size_t>((b * total_tokens + usable_tokens) * head_size_),
                    window_gate.begin() + static_cast<size_t>(b * usable_tokens * head_size_));
        }
        add_position_bias(window_gate, hca_pos, batch_size * usable_tokens, head_size_);
        new_entries = CompressWindows(window_kv, window_gate, batch_size, new_count, compress_rate_, head_size_,
                                      hca_norm.data(), rms_norm_epsilon_);
        const int64_t old_count = past_hca_entries->Shape().NumDimensions() == 4 ? past_hca_entries->Shape()[2]
                                                                                 : past_hca_entries->Shape()[1];
        const int64_t first_window_position = old_count * compress_rate_;
        for (int64_t b = 0; b < batch_size; ++b) {
          for (int64_t w = 0; w < new_count; ++w) {
            const int64_t pos = first_window_position + w * compress_rate_;
            ORT_RETURN_IF(pos < 0 || pos >= cos_rows, "compressor position out of range for cos/sin cache.");
            std::vector<float> entry(static_cast<size_t>(head_size_), 0.0f);
            const size_t offset = static_cast<size_t>((b * new_count + w) * head_size_);
            std::copy(new_entries.begin() + offset, new_entries.begin() + offset + static_cast<size_t>(head_size_),
                      entry.begin());
            ApplyInterleavedTrailingRope(entry, head_size_, rotary_dim_,
                                         cos_data.data() + pos * cos_width,
                                         sin_data.data() + pos * cos_width);
            std::copy(entry.begin(), entry.end(), new_entries.begin() + offset);
          }
        }
      }

      EntryState entry_state = ReadEntries(*past_hca_entries, batch_size, head_size_);
      auto past_entries = ReadEntryData(ToFloatVector<T>(*past_hca_entries), entry_state, batch_size, head_size_);
      AppendEntries(past_entries, entry_state.entries, new_entries, new_count, batch_size, head_size_);
      compressed_entries = past_entries;
      compressed_entry_count = entry_state.entries + new_count;
      if (5 < context->OutputCount()) {
        Tensor* entries_out = context->Output(5, EntryOutputShape(batch_size, compressed_entry_count,
                                                                  head_size_, entry_state.rank4));
        WriteFloatVector<T>(*entries_out, WriteEntryData(compressed_entries, batch_size, compressed_entry_count,
                                                         head_size_, entry_state.rank4));
      }
      return Status::OK();
    };

    auto make_csa_compressed = [&]() -> Status {
      const int64_t csa_width = 2 * head_size_;
      ORT_RETURN_IF_NOT(csa_kv_weight->Shape().NumDimensions() == 2 && csa_kv_weight->Shape()[0] == hidden_size &&
                            csa_kv_weight->Shape()[1] == csa_width,
                        "csa_kv_weight shape mismatch.");
      ORT_RETURN_IF_NOT(csa_gate_weight->Shape() == csa_kv_weight->Shape(), "csa_gate_weight shape mismatch.");
      ORT_RETURN_IF_NOT(csa_position_bias->Shape().NumDimensions() == 2 &&
                            csa_position_bias->Shape()[0] == compress_rate_ &&
                            csa_position_bias->Shape()[1] == csa_width,
                        "csa_position_bias shape mismatch.");
      ORT_RETURN_IF_NOT(csa_kv_norm_weight->Shape().NumDimensions() == 1 &&
                            csa_kv_norm_weight->Shape()[0] == head_size_,
                        "csa_kv_norm_weight shape mismatch.");

      const auto csa_kv_w = ToFloatVector<T>(*csa_kv_weight);
      const auto csa_gate_w = ToFloatVector<T>(*csa_gate_weight);
      const auto csa_pos = ToFloatVector<T>(*csa_position_bias);
      const auto csa_norm = ToFloatVector<T>(*csa_kv_norm_weight);
      auto kv = MakeRows(hidden_data, batch_size * sequence_length, hidden_size, csa_kv_w, csa_width);
      auto gate = MakeRows(hidden_data, batch_size * sequence_length, hidden_size, csa_gate_w, csa_width);

      int64_t total_tokens = 0;
      auto combined_kv = append_pending(*past_csa_pending_kv, kv, csa_width, total_tokens);
      auto combined_gate = append_pending(*past_csa_pending_gate, gate, csa_width, total_tokens);
      const int64_t usable_tokens = (total_tokens / compress_rate_) * compress_rate_;
      const int64_t new_count = usable_tokens / compress_rate_;
      write_pending_output(6, combined_kv, total_tokens, usable_tokens, csa_width);
      write_pending_output(7, combined_gate, total_tokens, usable_tokens, csa_width);

      std::vector<float> new_entries;
      if (new_count > 0) {
        std::vector<float> chunk_kv(static_cast<size_t>(batch_size * new_count * compress_rate_ * csa_width), 0.0f);
        std::vector<float> chunk_gate(chunk_kv.size(), 0.0f);
        for (int64_t b = 0; b < batch_size; ++b) {
          std::copy(combined_kv.begin() + static_cast<size_t>(b * total_tokens * csa_width),
                    combined_kv.begin() + static_cast<size_t>((b * total_tokens + usable_tokens) * csa_width),
                    chunk_kv.begin() + static_cast<size_t>(b * usable_tokens * csa_width));
          std::copy(combined_gate.begin() + static_cast<size_t>(b * total_tokens * csa_width),
                    combined_gate.begin() + static_cast<size_t>((b * total_tokens + usable_tokens) * csa_width),
                    chunk_gate.begin() + static_cast<size_t>(b * usable_tokens * csa_width));
        }
        add_position_bias(chunk_gate, csa_pos, batch_size * usable_tokens, csa_width);

        std::vector<float> window_kv(static_cast<size_t>(batch_size * new_count * 2 * compress_rate_ * head_size_),
                                     0.0f);
        std::vector<float> window_gate(window_kv.size(), -std::numeric_limits<float>::infinity());
        const auto prior_kv = ToFloatVector<T>(*past_csa_overlap_kv);
        const auto prior_gate = ToFloatVector<T>(*past_csa_overlap_gate);
        for (int64_t b = 0; b < batch_size; ++b) {
          for (int64_t w = 0; w < new_count; ++w) {
            for (int64_t r = 0; r < compress_rate_; ++r) {
              for (int64_t d = 0; d < head_size_; ++d) {
                const size_t dst_cb = static_cast<size_t>(((b * new_count + w) * 2 * compress_rate_ +
                                                           compress_rate_ + r) *
                                                              head_size_ +
                                                          d);
                const size_t src = static_cast<size_t>(((b * new_count + w) * compress_rate_ + r) * csa_width + d);
                const size_t src_cb = src + static_cast<size_t>(head_size_);
                window_kv[dst_cb] = chunk_kv[src_cb];
                window_gate[dst_cb] = chunk_gate[src_cb];
                const size_t dst_ca = static_cast<size_t>(((b * new_count + w) * 2 * compress_rate_ + r) *
                                                              head_size_ +
                                                          d);
                if (w == 0) {
                  const auto& overlap_shape = past_csa_overlap_kv->Shape();
                  if (overlap_shape.NumDimensions() == 3 && overlap_shape[1] == compress_rate_) {
                    const size_t overlap = static_cast<size_t>((b * compress_rate_ + r) * head_size_ + d);
                    window_kv[dst_ca] = prior_kv[overlap];
                    window_gate[dst_ca] = prior_gate[overlap];
                  }
                } else {
                  const size_t prev = static_cast<size_t>(((b * new_count + w - 1) * compress_rate_ + r) *
                                                              csa_width +
                                                          d);
                  window_kv[dst_ca] = chunk_kv[prev];
                  window_gate[dst_ca] = chunk_gate[prev];
                }
              }
            }
          }
        }
        new_entries = CompressWindows(window_kv, window_gate, batch_size, new_count, 2 * compress_rate_,
                                      head_size_, csa_norm.data(), rms_norm_epsilon_);
        const int64_t old_count = past_csa_entries->Shape().NumDimensions() == 4 ? past_csa_entries->Shape()[2]
                                                                                 : past_csa_entries->Shape()[1];
        for (int64_t b = 0; b < batch_size; ++b) {
          for (int64_t w = 0; w < new_count; ++w) {
            const int64_t pos = old_count * compress_rate_ + w * compress_rate_;
            ORT_RETURN_IF(pos < 0 || pos >= cos_rows, "compressor position out of range for cos/sin cache.");
            std::vector<float> entry(static_cast<size_t>(head_size_), 0.0f);
            const size_t offset = static_cast<size_t>((b * new_count + w) * head_size_);
            std::copy(new_entries.begin() + offset, new_entries.begin() + offset + static_cast<size_t>(head_size_),
                      entry.begin());
            ApplyInterleavedTrailingRope(entry, head_size_, rotary_dim_,
                                         cos_data.data() + pos * cos_width,
                                         sin_data.data() + pos * cos_width);
            std::copy(entry.begin(), entry.end(), new_entries.begin() + offset);
          }
        }

        if (context->OutputCount() > 10) {
          std::vector<float> overlap_kv(static_cast<size_t>(batch_size * compress_rate_ * head_size_), 0.0f);
          std::vector<float> overlap_gate(overlap_kv.size(), 0.0f);
          for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t r = 0; r < compress_rate_; ++r) {
              for (int64_t d = 0; d < head_size_; ++d) {
                const size_t dst = static_cast<size_t>((b * compress_rate_ + r) * head_size_ + d);
                const size_t src = static_cast<size_t>(((b * new_count + new_count - 1) * compress_rate_ + r) *
                                                           csa_width +
                                                       d);
                overlap_kv[dst] = chunk_kv[src];
                overlap_gate[dst] = chunk_gate[src];
              }
            }
          }
          Tensor* out_kv = context->Output(9, TensorShape({batch_size, compress_rate_, head_size_}));
          Tensor* out_gate = context->Output(10, TensorShape({batch_size, compress_rate_, head_size_}));
          WriteFloatVector<T>(*out_kv, overlap_kv);
          WriteFloatVector<T>(*out_gate, overlap_gate);
        }
      } else {
        if (context->OutputCount() > 10) {
          Tensor* out_kv = context->Output(9, past_csa_overlap_kv->Shape());
          Tensor* out_gate = context->Output(10, past_csa_overlap_gate->Shape());
          CopyTensor<T>(*past_csa_overlap_kv, *out_kv);
          CopyTensor<T>(*past_csa_overlap_gate, *out_gate);
        }
      }

      EntryState entry_state = ReadEntries(*past_csa_entries, batch_size, head_size_);
      auto past_entries = ReadEntryData(ToFloatVector<T>(*past_csa_entries), entry_state, batch_size, head_size_);
      AppendEntries(past_entries, entry_state.entries, new_entries, new_count, batch_size, head_size_);
      compressed_entries = past_entries;
      compressed_entry_count = entry_state.entries + new_count;
      if (8 < context->OutputCount()) {
        Tensor* entries_out = context->Output(8, EntryOutputShape(batch_size, compressed_entry_count,
                                                                  head_size_, entry_state.rank4));
        WriteFloatVector<T>(*entries_out, WriteEntryData(compressed_entries, batch_size, compressed_entry_count,
                                                         head_size_, entry_state.rank4));
      }

      // Keep the indexer states coherent even when the functional CPU path uses a deterministic
      // causal top-k over the compressed CSA entries.
      if (context->OutputCount() > 15) {
        Tensor* out = context->Output(11, past_index_pending_kv->Shape());
        CopyTensor<T>(*past_index_pending_kv, *out);
        out = context->Output(12, past_index_pending_gate->Shape());
        CopyTensor<T>(*past_index_pending_gate, *out);
        out = context->Output(13, past_index_entries->Shape());
        CopyTensor<T>(*past_index_entries, *out);
        out = context->Output(14, past_index_overlap_kv->Shape());
        CopyTensor<T>(*past_index_overlap_kv, *out);
        out = context->Output(15, past_index_overlap_gate->Shape());
        CopyTensor<T>(*past_index_overlap_gate, *out);
      }

      csa_selected_topk = std::min(index_topk_, compressed_entry_count);
      csa_selected_indices.assign(static_cast<size_t>(batch_size * sequence_length * csa_selected_topk), -1);
      for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < sequence_length; ++s) {
          const int64_t threshold = (position_data[b * sequence_length + s] + 1) / compress_rate_;
          const int64_t visible = std::min(threshold, compressed_entry_count);
          const int64_t first = std::max<int64_t>(0, visible - csa_selected_topk);
          for (int64_t k = 0; k < std::min(csa_selected_topk, visible); ++k) {
            csa_selected_indices[static_cast<size_t>((b * sequence_length + s) * csa_selected_topk + k)] = first + k;
          }
        }
      }
      return Status::OK();
    };

    if (attention_mode_ == DeepSeekV4AttentionMode::kHca) {
      ORT_RETURN_IF_ERROR(make_hca_compressed());
    } else if (attention_mode_ == DeepSeekV4AttentionMode::kCsa) {
      ORT_RETURN_IF_ERROR(make_csa_compressed());
    }

    auto* present_key_data = present_key->MutableData<T>();
    auto* present_value_data = present_value->MutableData<T>();
    auto* output_data = output->MutableData<T>();

    std::vector<float> q_a_row;
    std::vector<float> q_full;
    std::vector<float> kv_row;
    std::vector<float> key_head(static_cast<size_t>(head_size_), 0.0f);
    std::vector<float> value_head(static_cast<size_t>(head_size_), 0.0f);
    std::vector<float> context_heads(static_cast<size_t>(hidden_size), 0.0f);
    std::vector<float> o_a_row;
    std::vector<float> o_final;
    std::vector<float> logits;
    std::vector<float> exp_values;

    for (int64_t b = 0; b < batch_size; ++b) {
      ORT_RETURN_IF(seqlens_data[b] < 0, "seqlens_k values must be non-negative.");
      int64_t seq_len_k = static_cast<int64_t>(seqlens_data[b]);

      for (int64_t s = 0; s < sequence_length; ++s) {
        const int64_t hidden_offset = (b * sequence_length + s) * hidden_size;
        const float* hidden_row = hidden_data.data() + hidden_offset;

        MatMulRowMajor(hidden_row, 1, hidden_size, q_a_weight_data.data(), q_lora_rank_, q_a_row);
        ApplyRmsNorm(q_a_row, q_a_norm_weight_data.data(), q_lora_rank_, rms_norm_epsilon_);
        MatMulRowMajor(q_a_row.data(), 1, q_lora_rank_, q_b_weight_data.data(), hidden_size, q_full);

        MatMulRowMajor(hidden_row, 1, hidden_size, kv_weight_data.data(), kv_width, kv_row);
        ApplyRmsNorm(kv_row, kv_norm_weight_data.data(), kv_width, rms_norm_epsilon_);

        std::copy(kv_row.begin(), kv_row.begin() + head_size_, key_head.begin());
        std::copy(kv_row.begin() + head_size_, kv_row.begin() + 2 * head_size_, value_head.begin());

        const int64_t position = position_data[b * sequence_length + s];
        ORT_RETURN_IF(position < 0 || position >= cos_rows, "position id out of range for cos/sin cache.");
        const float* cos_row = cos_data.data() + position * cos_width;
        const float* sin_row = sin_data.data() + position * cos_width;

        for (int64_t h = 0; h < num_heads_; ++h) {
          std::vector<float> q_head(static_cast<size_t>(head_size_), 0.0f);
          const int64_t q_offset = h * head_size_;
          std::copy(q_full.begin() + q_offset, q_full.begin() + q_offset + head_size_, q_head.begin());
          ApplyInterleavedTrailingRope(q_head, head_size_, rotary_dim_, cos_row, sin_row);
          std::copy(q_head.begin(), q_head.end(), q_full.begin() + q_offset);
        }
        ApplyInterleavedTrailingRope(key_head, head_size_, rotary_dim_, cos_row, sin_row);

        int64_t cache_position = seq_len_k;
        if (cache_position >= cache_capacity) {
          const int64_t shift_count = cache_capacity - 1;
          if (shift_count > 0) {
            const int64_t batch_offset = ((b * kv_num_heads_) * cache_capacity) * head_size_;
            std::memmove(present_key_data + batch_offset,
                         present_key_data + batch_offset + head_size_,
                         static_cast<size_t>(shift_count * head_size_) * sizeof(T));
            std::memmove(present_value_data + batch_offset,
                         present_value_data + batch_offset + head_size_,
                         static_cast<size_t>(shift_count * head_size_) * sizeof(T));
          }
          cache_position = cache_capacity - 1;
          seq_len_k = cache_capacity - 1;
        }

        const int64_t cache_base = ((b * kv_num_heads_) * cache_capacity + cache_position) * head_size_;
        for (int64_t d = 0; d < head_size_; ++d) {
          present_key_data[cache_base + d] = FromFloat<T>(key_head[static_cast<size_t>(d)]);
          present_value_data[cache_base + d] = FromFloat<T>(value_head[static_cast<size_t>(d)]);
        }

        const int64_t query_total_index = seq_len_k;
        const int64_t available_length = std::min(seq_len_k + 1, cache_capacity);
        const int64_t attended_length = std::min(local_window_size_, available_length);
        const int64_t cache_start = available_length - attended_length;
        const int64_t key_total_start = query_total_index - available_length + 1 + cache_start;
        std::vector<int64_t> compressed_indices;
        if (attention_mode_ == DeepSeekV4AttentionMode::kHca && compressed_entry_count > 0) {
          const int64_t visible = std::min((position_data[b * sequence_length + s] + 1) / compress_rate_,
                                           compressed_entry_count);
          compressed_indices.reserve(static_cast<size_t>(visible));
          for (int64_t i = 0; i < visible; ++i) {
            compressed_indices.push_back(i);
          }
        } else if (attention_mode_ == DeepSeekV4AttentionMode::kCsa && csa_selected_topk > 0) {
          for (int64_t i = 0; i < csa_selected_topk; ++i) {
            const int64_t index =
                csa_selected_indices[static_cast<size_t>((b * sequence_length + s) * csa_selected_topk + i)];
            if (index >= 0 && index < compressed_entry_count) {
              compressed_indices.push_back(index);
            }
          }
        }
        const int64_t total_attended_length = attended_length + static_cast<int64_t>(compressed_indices.size());

        std::fill(context_heads.begin(), context_heads.end(), 0.0f);

        for (int64_t h = 0; h < num_heads_; ++h) {
          const int64_t q_offset = h * head_size_;
          const float* q_head = q_full.data() + q_offset;

          logits.assign(static_cast<size_t>(total_attended_length), 0.0f);
          exp_values.assign(static_cast<size_t>(total_attended_length), 0.0f);

          float max_logit = -std::numeric_limits<float>::infinity();
          for (int64_t i = 0; i < attended_length; ++i) {
            const int64_t cache_idx = cache_start + i;
            const int64_t key_offset = ((b * kv_num_heads_) * cache_capacity + cache_idx) * head_size_;

            float score = 0.0f;
            for (int64_t d = 0; d < head_size_; ++d) {
              score += q_head[d] * ToFloat(present_key_data[key_offset + d]);
            }
            score *= scale_;

            if (attention_bias != nullptr) {
              const int64_t key_total_index = key_total_start + i;
              score += ReadAttentionBias<T>(attention_bias, b, h, s, key_total_index);
            }

            logits[static_cast<size_t>(i)] = score;
            max_logit = std::max(max_logit, score);
          }
          for (size_t ci = 0; ci < compressed_indices.size(); ++ci) {
            const int64_t entry_index = compressed_indices[ci];
            const int64_t entry_offset = (b * compressed_entry_count + entry_index) * head_size_;
            float score = 0.0f;
            for (int64_t d = 0; d < head_size_; ++d) {
              score += q_head[d] * compressed_entries[static_cast<size_t>(entry_offset + d)];
            }
            score *= scale_;
            const int64_t logit_index = attended_length + static_cast<int64_t>(ci);
            logits[static_cast<size_t>(logit_index)] = score;
            max_logit = std::max(max_logit, score);
          }

          const float sink_logit = head_sink_data[head_sink_count == 1 ? 0 : h];
          max_logit = std::max(max_logit, sink_logit);

          float exp_sum = std::exp(sink_logit - max_logit);
          for (int64_t i = 0; i < total_attended_length; ++i) {
            const float e = std::exp(logits[static_cast<size_t>(i)] - max_logit);
            exp_values[static_cast<size_t>(i)] = e;
            exp_sum += e;
          }

          for (int64_t i = 0; i < attended_length; ++i) {
            const float weight = exp_values[static_cast<size_t>(i)] / exp_sum;
            const int64_t cache_idx = cache_start + i;
            const int64_t value_offset = ((b * kv_num_heads_) * cache_capacity + cache_idx) * head_size_;
            for (int64_t d = 0; d < head_size_; ++d) {
              context_heads[static_cast<size_t>(q_offset + d)] +=
                  weight * ToFloat(present_value_data[value_offset + d]);
            }
          }
          for (size_t ci = 0; ci < compressed_indices.size(); ++ci) {
            const float weight = exp_values[static_cast<size_t>(attended_length + ci)] / exp_sum;
            const int64_t entry_index = compressed_indices[ci];
            const int64_t entry_offset = (b * compressed_entry_count + entry_index) * head_size_;
            for (int64_t d = 0; d < head_size_; ++d) {
              context_heads[static_cast<size_t>(q_offset + d)] +=
                  weight * compressed_entries[static_cast<size_t>(entry_offset + d)];
            }
          }
        }

        MatMulRowMajor(context_heads.data(), 1, hidden_size, o_a_weight_data.data(), o_groups_ * o_lora_rank_, o_a_row);
        MatMulRowMajor(o_a_row.data(), 1, o_groups_ * o_lora_rank_, o_b_weight_data.data(), hidden_size, o_final);

        for (int64_t i = 0; i < hidden_size; ++i) {
          output_data[hidden_offset + i] = FromFloat<T>(o_final[static_cast<size_t>(i)]);
        }

        ++seq_len_k;
      }
    }

    return Status::OK();
  }

 private:
  int64_t num_heads_{};
  int64_t head_size_{};
  int64_t kv_num_heads_{};
  int64_t q_lora_rank_{};
  int64_t o_groups_{};
  int64_t o_lora_rank_{};
  int64_t rotary_dim_{};
  int64_t local_window_size_{};
  int64_t compress_rate_{};
  int64_t index_topk_{};
  int64_t index_num_heads_{};
  int64_t index_head_dim_{};
  float rms_norm_epsilon_{};
  float scale_{};
  DeepSeekV4AttentionMode attention_mode_{};
};

}  // namespace deepseek_v4_attention_impl

#define REGISTER_KERNEL_TYPED(T)                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      DeepSeekV4Attention,                                                  \
      kMSDomain,                                                            \
      1,                                                                    \
      T,                                                                    \
      kCpuExecutionProvider,                                                \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>())      \
          .TypeConstraint("P", DataTypeImpl::GetTensorType<int64_t>()),     \
      deepseek_v4_attention_impl::DeepSeekV4Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime
