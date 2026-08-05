// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/deepseek_v4_compression_common.h"

namespace onnxruntime {
namespace contrib {
namespace deepseek_v4_attention_impl {

template <typename T>
class HeavilyCompressedAttention final : public OpKernel {
 public:
  explicit HeavilyCompressedAttention(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr("compress_rate", &compress_rate_).IsOK() && compress_rate_ > 0,
                "HeavilyCompressedAttention: compress_rate must be greater than zero.");
    ORT_ENFORCE(info.GetAttr("rotary_dim", &rotary_dim_).IsOK() && rotary_dim_ > 0 && rotary_dim_ % 2 == 0,
                "HeavilyCompressedAttention: rotary_dim must be a positive even value.");
    rms_norm_epsilon_ = info.GetAttrOrDefault<float>("rms_norm_epsilon", 1e-6f);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* hidden = context->Input<Tensor>(0);
    const Tensor* position_ids = context->Input<Tensor>(1);
    const Tensor* cos_cache = context->Input<Tensor>(2);
    const Tensor* sin_cache = context->Input<Tensor>(3);
    const Tensor* kv_weight = context->Input<Tensor>(4);
    const Tensor* gate_weight = context->Input<Tensor>(5);
    const Tensor* position_bias = context->Input<Tensor>(6);
    const Tensor* norm_weight = context->Input<Tensor>(7);
    const Tensor* past_pending_kv = context->Input<Tensor>(8);
    const Tensor* past_pending_gate = context->Input<Tensor>(9);
    const Tensor* past_entries = context->Input<Tensor>(10);

    ORT_RETURN_IF_NOT(hidden && position_ids && cos_cache && sin_cache && kv_weight && gate_weight &&
                          position_bias && norm_weight && past_pending_kv && past_pending_gate && past_entries,
                      "HeavilyCompressedAttention: all inputs are required.");
    const auto& hidden_shape = hidden->Shape();
    ORT_RETURN_IF_NOT(hidden_shape.NumDimensions() == 3, "hidden_states must have rank 3.");
    const int64_t batch = hidden_shape[0];
    const int64_t sequence = hidden_shape[1];
    const int64_t hidden_size = hidden_shape[2];
    ORT_RETURN_IF_NOT(position_ids->Shape() == TensorShape({batch, sequence}),
                      "position_ids must have shape (B, S).");
    ORT_RETURN_IF_NOT(cos_cache->Shape() == sin_cache->Shape() && cos_cache->Shape().NumDimensions() == 2,
                      "cos_cache and sin_cache must have the same rank-2 shape.");
    ORT_RETURN_IF_NOT(kv_weight->Shape().NumDimensions() == 2 && kv_weight->Shape()[0] == hidden_size,
                      "kv_weight must have shape (D, H).");
    const int64_t head_size = kv_weight->Shape()[1];
    ORT_RETURN_IF_NOT(gate_weight->Shape() == kv_weight->Shape(),
                      "gate_weight must have the same shape as kv_weight.");
    ORT_RETURN_IF_NOT(position_bias->Shape() == TensorShape({compress_rate_, head_size}),
                      "position_bias must have shape (compress_rate, H).");
    ORT_RETURN_IF_NOT(norm_weight->Shape() == TensorShape({head_size}),
                      "norm_weight must have shape (H).");
    ORT_RETURN_IF_NOT(rotary_dim_ <= head_size && cos_cache->Shape()[1] * 2 >= rotary_dim_,
                      "rotary dimensions are incompatible with H and the caches.");
    ORT_RETURN_IF_NOT(past_pending_kv->Shape().NumDimensions() == 3 &&
                          past_pending_kv->Shape()[0] == batch && past_pending_kv->Shape()[2] == head_size &&
                          past_pending_gate->Shape() == past_pending_kv->Shape(),
                      "pending states must have shape (B, P, H).");

    EntryState entry_state = ReadEntries(*past_entries, batch, head_size);
    const int64_t pending_count = past_pending_kv->Shape()[1];
    const int64_t total_count = pending_count + sequence;
    const int64_t usable_count = total_count / compress_rate_ * compress_rate_;
    const int64_t new_entry_count = usable_count / compress_rate_;
    const int64_t remaining_count = total_count - usable_count;
    const int64_t updated_entry_count = entry_state.entries + new_entry_count;
    ORT_RETURN_IF(new_entry_count > 0 &&
                      (updated_entry_count - 1) * compress_rate_ >= cos_cache->Shape()[0],
                  "compressed entry position is outside the RoPE cache.");

    const auto hidden_data = ToFloatVector<T>(*hidden);
    const auto kv_weight_data = ToFloatVector<T>(*kv_weight);
    const auto gate_weight_data = ToFloatVector<T>(*gate_weight);
    const auto bias_data = ToFloatVector<T>(*position_bias);
    const auto norm_data = ToFloatVector<T>(*norm_weight);
    const auto cos_data = ToFloatVector<T>(*cos_cache);
    const auto sin_data = ToFloatVector<T>(*sin_cache);
    auto current_kv = MakeRows(hidden_data, batch * sequence, hidden_size, kv_weight_data, head_size);
    auto current_gate = MakeRows(hidden_data, batch * sequence, hidden_size, gate_weight_data, head_size);

    auto combine = [&](const Tensor& pending, const std::vector<float>& current) {
      const auto pending_data = ToFloatVector<T>(pending);
      std::vector<float> combined(static_cast<size_t>(batch * total_count * head_size));
      for (int64_t b = 0; b < batch; ++b) {
        std::copy_n(pending_data.begin() + b * pending_count * head_size,
                    pending_count * head_size, combined.begin() + b * total_count * head_size);
        std::copy_n(current.begin() + b * sequence * head_size,
                    sequence * head_size,
                    combined.begin() + (b * total_count + pending_count) * head_size);
      }
      return combined;
    };
    auto combined_kv = combine(*past_pending_kv, current_kv);
    auto combined_gate = combine(*past_pending_gate, current_gate);

    std::vector<float> window_kv(static_cast<size_t>(batch * usable_count * head_size));
    std::vector<float> window_gate(window_kv.size());
    for (int64_t b = 0; b < batch; ++b) {
      std::copy_n(combined_kv.begin() + b * total_count * head_size, usable_count * head_size,
                  window_kv.begin() + b * usable_count * head_size);
      std::copy_n(combined_gate.begin() + b * total_count * head_size, usable_count * head_size,
                  window_gate.begin() + b * usable_count * head_size);
    }
    for (int64_t row = 0; row < batch * usable_count; ++row) {
      for (int64_t d = 0; d < head_size; ++d) {
        window_gate[static_cast<size_t>(row * head_size + d)] +=
            bias_data[static_cast<size_t>((row % compress_rate_) * head_size + d)];
      }
    }
    auto new_entries = CompressWindows(window_kv, window_gate, batch, new_entry_count, compress_rate_,
                                       head_size, norm_data.data(), rms_norm_epsilon_);
    const int64_t cache_width = cos_cache->Shape()[1];
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t entry = 0; entry < new_entry_count; ++entry) {
        const int64_t position = (entry_state.entries + entry) * compress_rate_;
        std::vector<float> row(static_cast<size_t>(head_size));
        const size_t offset = static_cast<size_t>((b * new_entry_count + entry) * head_size);
        std::copy_n(new_entries.begin() + offset, head_size, row.begin());
        ApplyInterleavedTrailingRope(row, head_size, rotary_dim_, cos_data.data() + position * cache_width,
                                     sin_data.data() + position * cache_width);
        std::copy(row.begin(), row.end(), new_entries.begin() + offset);
      }
    }

    auto entries = ReadEntryData(ToFloatVector<T>(*past_entries), entry_state, batch, head_size);
    AppendEntries(entries, entry_state.entries, new_entries, new_entry_count, batch, head_size);
    const TensorShape entries_shape({batch, 1, updated_entry_count, head_size});
    for (int output_index : {0, 4}) {
      Tensor* output = context->Output(output_index, entries_shape);
      WriteFloatVector<T>(*output, entries);
    }

    Tensor* block_bias = context->Output(1, TensorShape({batch, 1, sequence, updated_entry_count}));
    std::vector<float> block_bias_data(static_cast<size_t>(batch * sequence * updated_entry_count),
                                       -std::numeric_limits<float>::infinity());
    const int64_t* positions = position_ids->Data<int64_t>();
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < sequence; ++s) {
        const int64_t visible = std::min(updated_entry_count,
                                         (positions[b * sequence + s] + 1) / compress_rate_);
        std::fill_n(block_bias_data.begin() + (b * sequence + s) * updated_entry_count, visible, 0.0f);
      }
    }
    WriteFloatVector<T>(*block_bias, block_bias_data);

    for (int output_index = 2; output_index <= 3; ++output_index) {
      const auto& combined = output_index == 2 ? combined_kv : combined_gate;
      std::vector<float> remaining(static_cast<size_t>(batch * remaining_count * head_size));
      for (int64_t b = 0; b < batch; ++b) {
        std::copy_n(combined.begin() + (b * total_count + usable_count) * head_size,
                    remaining_count * head_size, remaining.begin() + b * remaining_count * head_size);
      }
      Tensor* output = context->Output(output_index, TensorShape({batch, remaining_count, head_size}));
      WriteFloatVector<T>(*output, remaining);
    }
    return Status::OK();
  }

 private:
  int64_t compress_rate_{};
  int64_t rotary_dim_{};
  float rms_norm_epsilon_{};
};

}  // namespace deepseek_v4_attention_impl

#define REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      HeavilyCompressedAttention, kMSDomain, 1, T, kCpuExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()), \
      deepseek_v4_attention_impl::HeavilyCompressedAttention<T>);

REGISTER_KERNEL(float)
REGISTER_KERNEL(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime
