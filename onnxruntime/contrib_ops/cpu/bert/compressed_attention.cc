// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/deepseek_v4_compression_common.h"

namespace onnxruntime {
namespace contrib {
namespace deepseek_v4_attention_impl {

template <typename T>
class CompressedAttention final : public OpKernel {
 public:
  explicit CompressedAttention(const OpKernelInfo& info) : OpKernel(info) {
    has_scale_ = info.GetAttr("scale", &scale_).IsOK();
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* query = context->Input<Tensor>(0);
    const Tensor* local_kv = context->Input<Tensor>(1);
    const Tensor* compressed_kv = context->Input<Tensor>(2);
    const Tensor* attention_bias = context->Input<Tensor>(3);
    const Tensor* selected_indices = context->Input<Tensor>(4);
    const Tensor* head_sink = context->Input<Tensor>(5);
    ORT_RETURN_IF_NOT(query && local_kv && head_sink, "query, local_kv, and head_sink are required.");
    const auto& query_shape = query->Shape();
    ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 4, "query must have shape (B, N, S, H).");
    const int64_t batch = query_shape[0];
    const int64_t num_heads = query_shape[1];
    const int64_t sequence = query_shape[2];
    const int64_t head_size = query_shape[3];
    ORT_RETURN_IF_NOT(local_kv->Shape().NumDimensions() == 4 && local_kv->Shape()[0] == batch &&
                          local_kv->Shape()[1] == 1 && local_kv->Shape()[3] == head_size,
                      "local_kv must have shape (B, 1, L, H).");
    const int64_t local_count = local_kv->Shape()[2];
    int64_t compressed_count = 0;
    if (compressed_kv) {
      ORT_RETURN_IF_NOT(compressed_kv->Shape().NumDimensions() == 4 && compressed_kv->Shape()[0] == batch &&
                            compressed_kv->Shape()[1] == 1 && compressed_kv->Shape()[3] == head_size,
                        "compressed_kv must have shape (B, 1, E, H).");
      compressed_count = compressed_kv->Shape()[2];
    }
    if (selected_indices) {
      ORT_RETURN_IF_NOT(compressed_kv && selected_indices->Shape().NumDimensions() == 3 &&
                            selected_indices->Shape()[0] == batch && selected_indices->Shape()[1] == sequence,
                        "selected_indices must have shape (B, S, K) and requires compressed_kv.");
    }
    const int64_t sink_count = head_sink->Shape().NumDimensions() == 0 ? 1 : head_sink->Shape()[0];
    ORT_RETURN_IF_NOT(head_sink->Shape().NumDimensions() <= 1 && (sink_count == 1 || sink_count == num_heads),
                      "head_sink must be scalar or have shape (N).");
    if (attention_bias) {
      ORT_RETURN_IF_NOT(attention_bias->Shape().NumDimensions() == 4,
                        "attention_bias must have rank 4.");
      const int64_t expected[] = {batch, num_heads, sequence, local_count + compressed_count};
      for (int dimension = 0; dimension < 4; ++dimension) {
        ORT_RETURN_IF_NOT(attention_bias->Shape()[dimension] == 1 || attention_bias->Shape()[dimension] == expected[dimension],
                          "attention_bias is not broadcastable to (B, N, S, L + E).");
      }
    }

    const auto query_data = ToFloatVector<T>(*query);
    const auto local_data = ToFloatVector<T>(*local_kv);
    const auto compressed_data = compressed_kv ? ToFloatVector<T>(*compressed_kv) : std::vector<float>{};
    const auto bias_data = attention_bias ? ToFloatVector<T>(*attention_bias) : std::vector<float>{};
    const auto sink_data = ToFloatVector<T>(*head_sink);
    const int64_t selected_count = selected_indices ? selected_indices->Shape()[2] : compressed_count;
    const int64_t* selected_data = selected_indices ? selected_indices->Data<int64_t>() : nullptr;
    const float scale = has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size));
    std::vector<float> output(query_data.size(), 0.0f);
    auto read_bias = [&](int64_t b, int64_t n, int64_t s, int64_t key) {
      if (!attention_bias) {
        return 0.0f;
      }
      const auto& shape = attention_bias->Shape();
      const int64_t indices[] = {b, n, s, key};
      int64_t offset = 0;
      for (int dimension = 0; dimension < 4; ++dimension) {
        offset = offset * shape[dimension] + (shape[dimension] == 1 ? 0 : indices[dimension]);
      }
      return bias_data[static_cast<size_t>(offset)];
    };

    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t n = 0; n < num_heads; ++n) {
        for (int64_t s = 0; s < sequence; ++s) {
          std::vector<int64_t> compressed_indices;
          for (int64_t index = 0; index < selected_count; ++index) {
            const int64_t entry = selected_data
                                      ? selected_data[(b * sequence + s) * selected_count + index]
                                      : index;
            ORT_RETURN_IF(entry < -1 || entry >= compressed_count, "selected index is outside compressed_kv.");
            if (entry >= 0) {
              compressed_indices.push_back(entry);
            }
          }
          const int64_t key_count = local_count + static_cast<int64_t>(compressed_indices.size());
          std::vector<float> logits(static_cast<size_t>(key_count));
          const size_t query_offset = static_cast<size_t>(((b * num_heads + n) * sequence + s) * head_size);
          float max_logit = sink_data[static_cast<size_t>(sink_count == 1 ? 0 : n)];
          for (int64_t key = 0; key < local_count; ++key) {
            float dot = 0.0f;
            const size_t key_offset = static_cast<size_t>((b * local_count + key) * head_size);
            for (int64_t d = 0; d < head_size; ++d) {
              dot += query_data[query_offset + d] * local_data[key_offset + d];
            }
            logits[static_cast<size_t>(key)] = dot * scale + read_bias(b, n, s, key);
            max_logit = std::max(max_logit, logits[static_cast<size_t>(key)]);
          }
          for (size_t index = 0; index < compressed_indices.size(); ++index) {
            const int64_t entry = compressed_indices[index];
            float dot = 0.0f;
            const size_t key_offset = static_cast<size_t>((b * compressed_count + entry) * head_size);
            for (int64_t d = 0; d < head_size; ++d) {
              dot += query_data[query_offset + d] * compressed_data[key_offset + d];
            }
            const size_t logit_index = static_cast<size_t>(local_count) + index;
            logits[logit_index] = dot * scale + read_bias(b, n, s, local_count + entry);
            max_logit = std::max(max_logit, logits[logit_index]);
          }
          float denominator = std::exp(sink_data[static_cast<size_t>(sink_count == 1 ? 0 : n)] - max_logit);
          for (float& logit : logits) {
            logit = std::exp(logit - max_logit);
            denominator += logit;
          }
          for (int64_t key = 0; key < local_count; ++key) {
            const size_t value_offset = static_cast<size_t>((b * local_count + key) * head_size);
            for (int64_t d = 0; d < head_size; ++d) {
              output[query_offset + d] += logits[static_cast<size_t>(key)] / denominator * local_data[value_offset + d];
            }
          }
          for (size_t index = 0; index < compressed_indices.size(); ++index) {
            const size_t value_offset = static_cast<size_t>((b * compressed_count + compressed_indices[index]) * head_size);
            for (int64_t d = 0; d < head_size; ++d) {
              output[query_offset + d] += logits[static_cast<size_t>(local_count) + index] / denominator *
                                                  compressed_data[value_offset + d];
            }
          }
        }
      }
    }
    WriteFloatVector<T>(*context->Output(0, query_shape), output);
    return Status::OK();
  }

 private:
  bool has_scale_{};
  float scale_{};
};

}  // namespace deepseek_v4_attention_impl

#define REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      CompressedAttention, kMSDomain, 1, T, kCpuExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", {DataTypeImpl::GetTensorType<int32_t>(), \
                                  DataTypeImpl::GetTensorType<int64_t>()}), \
      deepseek_v4_attention_impl::CompressedAttention<T>);

REGISTER_KERNEL(float)
REGISTER_KERNEL(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime
