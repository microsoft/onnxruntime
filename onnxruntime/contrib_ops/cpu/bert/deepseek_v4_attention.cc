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

namespace {

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
    }

    if (attention_mode_ == DeepSeekV4AttentionMode::kCsa) {
      int64_t index_topk = 0;
      int64_t index_num_heads = 0;
      int64_t index_head_dim = 0;
      ORT_ENFORCE(info.GetAttr("index_topk", &index_topk).IsOK() && index_topk > 0,
                  "DeepSeekV4Attention: index_topk must be provided and > 0 for csa mode.");
      ORT_ENFORCE(info.GetAttr("index_num_heads", &index_num_heads).IsOK() && index_num_heads > 0,
                  "DeepSeekV4Attention: index_num_heads must be provided and > 0 for csa mode.");
      ORT_ENFORCE(info.GetAttr("index_head_dim", &index_head_dim).IsOK() && index_head_dim > 0,
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

    ORT_RETURN_IF_NOT(hidden_states != nullptr && position_ids != nullptr &&
                          past_key != nullptr && past_value != nullptr &&
                          seqlens_k != nullptr && total_sequence_length != nullptr &&
                          cos_cache != nullptr && sin_cache != nullptr &&
                          q_a_weight != nullptr && q_a_norm_weight != nullptr &&
                          q_b_weight != nullptr && kv_weight != nullptr &&
                          kv_norm_weight != nullptr && o_a_weight != nullptr &&
                          o_b_weight != nullptr && head_sink != nullptr,
                      "DeepSeekV4Attention: required input is missing.");

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

    auto* present_key_data = present_key->MutableData<T>();
    auto* present_value_data = present_value->MutableData<T>();
    auto* output_data = output->MutableData<T>();

    const int64_t cos_rows = cos_cache->Shape()[0];
    const int64_t cos_width = cos_cache->Shape()[1];

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

        std::fill(context_heads.begin(), context_heads.end(), 0.0f);

        for (int64_t h = 0; h < num_heads_; ++h) {
          const int64_t q_offset = h * head_size_;
          const float* q_head = q_full.data() + q_offset;

          logits.assign(static_cast<size_t>(attended_length), 0.0f);
          exp_values.assign(static_cast<size_t>(attended_length), 0.0f);

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

          const float sink_logit = head_sink_data[head_sink_count == 1 ? 0 : h];
          max_logit = std::max(max_logit, sink_logit);

          float exp_sum = std::exp(sink_logit - max_logit);
          for (int64_t i = 0; i < attended_length; ++i) {
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
  float rms_norm_epsilon_{};
  float scale_{};
  DeepSeekV4AttentionMode attention_mode_{};
};

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
      DeepSeekV4Attention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

}  // namespace

}  // namespace contrib
}  // namespace onnxruntime
