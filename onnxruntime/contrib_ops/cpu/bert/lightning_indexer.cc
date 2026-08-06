// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/deepseek_v4_compression_common.h"

namespace onnxruntime {
namespace contrib {
namespace deepseek_v4_attention_impl {

template <typename T>
class LightningIndexer final : public OpKernel {
 public:
  explicit LightningIndexer(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr("compress_rate", &compress_rate_).IsOK() && compress_rate_ > 0,
                "LightningIndexer: compress_rate must be greater than zero.");
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads_).IsOK() && num_heads_ > 0,
                "LightningIndexer: num_heads must be greater than zero.");
    ORT_ENFORCE(info.GetAttr("head_size", &head_size_).IsOK() && head_size_ > 0,
                "LightningIndexer: head_size must be greater than zero.");
    ORT_ENFORCE(info.GetAttr("index_topk", &index_topk_).IsOK() && index_topk_ > 0,
                "LightningIndexer: index_topk must be greater than zero.");
    ORT_ENFORCE(info.GetAttr("rotary_dim", &rotary_dim_).IsOK() && rotary_dim_ > 0 && rotary_dim_ % 2 == 0,
                "LightningIndexer: rotary_dim must be a positive even value.");
    rms_norm_epsilon_ = info.GetAttrOrDefault<float>("rms_norm_epsilon", 1e-6f);
    entry_capacity_ = info.GetAttrOrDefault<int64_t>("entry_capacity", 0);
  }

  Status Compute(OpKernelContext* context) const override {
    for (int index = 0; index < 16; ++index) {
      ORT_RETURN_IF_NOT(context->Input<Tensor>(index) != nullptr,
                        "LightningIndexer: all inputs are required.");
    }
    const Tensor& hidden = *context->Input<Tensor>(0);
    const Tensor& q_residual = *context->Input<Tensor>(1);
    const auto& hidden_shape = hidden.Shape();
    ORT_RETURN_IF_NOT(hidden_shape.NumDimensions() == 3, "hidden_states must have rank 3.");
    const int64_t batch = hidden_shape[0];
    const int64_t sequence = hidden_shape[1];
    const int64_t hidden_size = hidden_shape[2];
    ORT_RETURN_IF_NOT(q_residual.Shape().NumDimensions() == 3 && q_residual.Shape()[0] == batch &&
                          q_residual.Shape()[1] == sequence,
                      "q_residual must have shape (B, S, R).");
    const int64_t q_rank = q_residual.Shape()[2];
    ORT_RETURN_IF_NOT(context->Input<Tensor>(5)->Shape() == TensorShape({hidden_size, 2 * head_size_}),
                      "kv_weight must have shape (D, 2 * head_size).");
    ORT_RETURN_IF_NOT(context->Input<Tensor>(9)->Shape() == TensorShape({q_rank, num_heads_ * head_size_}),
                      "q_weight must have shape (R, num_heads * head_size).");
    ORT_RETURN_IF_NOT(context->Input<Tensor>(10)->Shape() == TensorShape({hidden_size, num_heads_}),
                      "score_weight must have shape (D, num_heads).");
    ORT_RETURN_IF_NOT(context->Input<Tensor>(13)->Shape().NumDimensions() == 3,
                      "past_entries must have shape (B, E, head_size).");

    CompressorResult result;
    ORT_RETURN_IF_ERROR(RunCompressor<T>(
      hidden, *context->Input<Tensor>(3), *context->Input<Tensor>(4),
        *context->Input<Tensor>(5), *context->Input<Tensor>(6), *context->Input<Tensor>(7),
        *context->Input<Tensor>(8), *context->Input<Tensor>(11), *context->Input<Tensor>(12),
      *context->Input<Tensor>(13), context->Input<Tensor>(14), context->Input<Tensor>(15),
        compress_rate_, rotary_dim_, rms_norm_epsilon_, result,
        entry_capacity_, context->Input<Tensor>(2)));

    const bool fixed_mode = entry_capacity_ > 0;
    const Tensor& past_pending = *context->Input<Tensor>(11);
    const TensorShape pending_shape = fixed_mode
                                          ? past_pending.Shape()
                                          : TensorShape({batch, result.pending_count, 2 * head_size_});
    std::vector<float> pending_kv = result.pending_kv;
    std::vector<float> pending_gate = result.pending_gate;
    if (fixed_mode) {
      pending_kv.assign(static_cast<size_t>(past_pending.Shape().Size()), 0.0f);
      pending_gate.assign(pending_kv.size(), 0.0f);
      for (int64_t b = 0; b < batch; ++b) {
        const size_t source = static_cast<size_t>(b * result.pending_count * 2 * head_size_);
        const size_t destination = static_cast<size_t>(b * (compress_rate_ - 1) * 2 * head_size_);
        std::copy_n(result.pending_kv.begin() + source, result.pending_count * 2 * head_size_,
                    pending_kv.begin() + destination);
        std::copy_n(result.pending_gate.begin() + source, result.pending_count * 2 * head_size_,
                    pending_gate.begin() + destination);
      }
    }
    WriteFloatVector<T>(*context->Output(1, pending_shape), pending_kv);
    WriteFloatVector<T>(*context->Output(2, pending_shape), pending_gate);
    const TensorShape entries_shape = fixed_mode
                                          ? context->Input<Tensor>(13)->Shape()
                                          : EntryOutputShape(batch, result.entry_count, head_size_, result.entries_rank4);
    const auto entries = WriteEntryData(result.entries, batch, result.entry_count,
                                        head_size_, result.entries_rank4);
    WriteFloatVector<T>(*context->Output(3, entries_shape), entries);
    WriteFloatVector<T>(*context->Output(4, TensorShape({batch, compress_rate_, head_size_})), result.overlap_kv);
    WriteFloatVector<T>(*context->Output(5, TensorShape({batch, compress_rate_, head_size_})), result.overlap_gate);

    const auto q_data = ToFloatVector<T>(q_residual);
    const auto q_weight = ToFloatVector<T>(*context->Input<Tensor>(9));
    const auto hidden_data = ToFloatVector<T>(hidden);
    const auto score_weight = ToFloatVector<T>(*context->Input<Tensor>(10));
    const auto cos_data = ToFloatVector<T>(*context->Input<Tensor>(3));
    const auto sin_data = ToFloatVector<T>(*context->Input<Tensor>(4));
    auto queries = MakeRows(q_data, batch * sequence, q_rank, q_weight, num_heads_ * head_size_);
    auto head_weights = MakeRows(hidden_data, batch * sequence, hidden_size, score_weight, num_heads_);
    const Tensor& cos_cache = *context->Input<Tensor>(3);
    const int64_t cache_width = cos_cache.Shape()[1];
    const int64_t* positions = context->Input<Tensor>(2)->Data<int64_t>();
    const float dot_scale = 1.0f / std::sqrt(static_cast<float>(head_size_));
    const float head_scale = 1.0f / std::sqrt(static_cast<float>(num_heads_));
    Tensor* selected_indices = context->Output(0, TensorShape({batch, sequence, index_topk_}));
    int64_t* selected_data = selected_indices->MutableData<int64_t>();
    std::fill_n(selected_data, batch * sequence * index_topk_, int64_t{-1});
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < sequence; ++s) {
        const int64_t position = positions[b * sequence + s];
        ORT_RETURN_IF(position < 0 || position >= cos_cache.Shape()[0],
                      "position id is outside the RoPE cache.");
        for (int64_t head = 0; head < num_heads_; ++head) {
          const size_t offset = static_cast<size_t>(((b * sequence + s) * num_heads_ + head) * head_size_);
          std::vector<float> row(queries.begin() + offset, queries.begin() + offset + head_size_);
          ApplyInterleavedTrailingRope(row, head_size_, rotary_dim_, cos_data.data() + position * cache_width,
                                       sin_data.data() + position * cache_width);
          std::copy(row.begin(), row.end(), queries.begin() + offset);
        }
        const int64_t visible = std::min(result.entry_count, (position + 1) / compress_rate_);
        std::vector<std::pair<float, int64_t>> scores;
        scores.reserve(static_cast<size_t>(visible));
        for (int64_t entry = 0; entry < visible; ++entry) {
          float score = 0.0f;
          for (int64_t head = 0; head < num_heads_; ++head) {
            float dot = 0.0f;
            const size_t query_offset = static_cast<size_t>(((b * sequence + s) * num_heads_ + head) * head_size_);
            const size_t entry_offset = static_cast<size_t>((b * result.entry_count + entry) * head_size_);
            for (int64_t d = 0; d < head_size_; ++d) {
              dot += queries[query_offset + d] * result.entries[entry_offset + d];
            }
            const float projected_head_weight =
                head_weights[static_cast<size_t>((b * sequence + s) * num_heads_ + head)] * head_scale;
            score += projected_head_weight * std::max(0.0f, dot * dot_scale);
          }
          scores.emplace_back(score, entry);
        }
        std::sort(scores.begin(), scores.end(), [](const auto& left, const auto& right) {
          return left.first != right.first ? left.first > right.first : left.second < right.second;
        });
        for (int64_t rank = 0; rank < std::min<int64_t>(index_topk_, visible); ++rank) {
          selected_data[(b * sequence + s) * index_topk_ + rank] = scores[static_cast<size_t>(rank)].second;
        }
      }
    }
    return Status::OK();
  }

 private:
  int64_t compress_rate_{};
  int64_t num_heads_{};
  int64_t head_size_{};
  int64_t index_topk_{};
  int64_t rotary_dim_{};
  float rms_norm_epsilon_{};
  int64_t entry_capacity_{};
};

}  // namespace deepseek_v4_attention_impl

#define REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      LightningIndexer, kMSDomain, 1, T, kCpuExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", {DataTypeImpl::GetTensorType<int32_t>(), \
                                  DataTypeImpl::GetTensorType<int64_t>()}) \
          .MayInplace(11, 1)                                            \
          .MayInplace(12, 2)                                            \
          .MayInplace(13, 3)                                            \
          .MayInplace(14, 4)                                            \
          .MayInplace(15, 5),                                           \
      deepseek_v4_attention_impl::LightningIndexer<T>);

REGISTER_KERNEL(float)
REGISTER_KERNEL(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime
