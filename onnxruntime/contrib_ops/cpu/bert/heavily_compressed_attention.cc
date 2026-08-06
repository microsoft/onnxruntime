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
    entry_capacity_ = info.GetAttrOrDefault<int64_t>("entry_capacity", 0);
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
    ORT_RETURN_IF_NOT(position_ids->Shape() == TensorShape({batch, sequence}),
                      "position_ids must have shape (B, S).");

    const bool fixed_mode = entry_capacity_ > 0;
    CompressorResult result;
    ORT_RETURN_IF_ERROR(RunCompressor<T>(
        *hidden, *cos_cache, *sin_cache, *kv_weight, *gate_weight, *position_bias, *norm_weight,
        *past_pending_kv, *past_pending_gate, *past_entries, nullptr, nullptr,
        compress_rate_, rotary_dim_, rms_norm_epsilon_, result,
        entry_capacity_, position_ids));

    const int64_t head_size = norm_weight->Shape()[0];
    if (fixed_mode) {
      // Fixed mode: output shapes match input shapes.
      for (int output_index : {0, 4}) {
        Tensor* output = context->Output(output_index, past_entries->Shape());
        WriteFloatVector<T>(*output, result.entries);
      }
      Tensor* block_bias = context->Output(1, TensorShape({batch, 1, sequence, entry_capacity_}));
      std::vector<float> block_bias_data(static_cast<size_t>(batch * sequence * entry_capacity_),
                                         -std::numeric_limits<float>::infinity());
      const int64_t* positions = position_ids->Data<int64_t>();
      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < sequence; ++s) {
          const int64_t visible = std::min(entry_capacity_,
                                           (positions[b * sequence + s] + 1) / compress_rate_);
          std::fill_n(block_bias_data.begin() + (b * sequence + s) * entry_capacity_, visible, 0.0f);
        }
      }
      WriteFloatVector<T>(*block_bias, block_bias_data);
      // Pending output: fixed capacity compress_rate-1, zero-padded.
      for (int output_index = 2; output_index <= 3; ++output_index) {
        const auto& pending = output_index == 2 ? result.pending_kv : result.pending_gate;
        Tensor* output = context->Output(output_index, past_pending_kv->Shape());
        std::vector<float> padded(static_cast<size_t>(past_pending_kv->Shape().Size()), 0.0f);
        for (int64_t b = 0; b < batch; ++b) {
          std::copy_n(pending.begin() + b * result.pending_count * head_size,
                      result.pending_count * head_size,
                      padded.begin() + b * (compress_rate_ - 1) * head_size);
        }
        WriteFloatVector<T>(*output, padded);
      }
    } else {
      const TensorShape entries_shape({batch, 1, result.entry_count, head_size});
      for (int output_index : {0, 4}) {
        Tensor* output = context->Output(output_index, entries_shape);
        WriteFloatVector<T>(*output, result.entries);
      }
      Tensor* block_bias = context->Output(1, TensorShape({batch, 1, sequence, result.entry_count}));
      std::vector<float> block_bias_data(static_cast<size_t>(batch * sequence * result.entry_count),
                                         -std::numeric_limits<float>::infinity());
      const int64_t* positions = position_ids->Data<int64_t>();
      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < sequence; ++s) {
          const int64_t visible = std::min(result.entry_count,
                                           (positions[b * sequence + s] + 1) / compress_rate_);
          std::fill_n(block_bias_data.begin() + (b * sequence + s) * result.entry_count, visible, 0.0f);
        }
      }
      WriteFloatVector<T>(*block_bias, block_bias_data);
      for (int output_index = 2; output_index <= 3; ++output_index) {
        const auto& pending = output_index == 2 ? result.pending_kv : result.pending_gate;
        Tensor* output = context->Output(output_index, TensorShape({batch, result.pending_count, head_size}));
        WriteFloatVector<T>(*output, pending);
      }
    }
    return Status::OK();
  }

 private:
  int64_t compress_rate_{};
  int64_t rotary_dim_{};
  float rms_norm_epsilon_{};
  int64_t entry_capacity_{};
};

}  // namespace deepseek_v4_attention_impl

#define REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      HeavilyCompressedAttention, kMSDomain, 1, T, kCpuExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()) \
          .MayInplace(8, 2)                                             \
          .MayInplace(9, 3)                                             \
          .MayInplace(10, 4),                                           \
      deepseek_v4_attention_impl::HeavilyCompressedAttention<T>);

REGISTER_KERNEL(float)
REGISTER_KERNEL(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime
