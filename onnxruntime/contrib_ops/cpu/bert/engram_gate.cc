// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/engram_gate.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "contrib_ops/cpu/bert/kernel_helper.h"
#include "core/common/narrow.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

#define REGISTER_ENGRAM_GATE_TYPED(T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      EngramGate,                                                 \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      EngramGate<T>);

REGISTER_ENGRAM_GATE_TYPED(float)
REGISTER_ENGRAM_GATE_TYPED(MLFloat16)

#undef REGISTER_ENGRAM_GATE_TYPED

template <typename T>
EngramGate<T>::EngramGate(const OpKernelInfo& info) : OpKernel(info) {
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

template <typename T>
Status EngramGate<T>::Compute(OpKernelContext* context) const {
  const Tensor* embeddings = context->Input<Tensor>(0);
  const Tensor* hidden_states = context->Input<Tensor>(1);
  const Tensor* key_weight = context->Input<Tensor>(2);
  const Tensor* key_bias = context->Input<Tensor>(3);
  const Tensor* value_weight = context->Input<Tensor>(4);
  const Tensor* value_bias = context->Input<Tensor>(5);
  const Tensor* key_norm_scale = context->Input<Tensor>(6);
  const Tensor* query_norm_scale = context->Input<Tensor>(7);
  const Tensor* conv_norm_scale = context->Input<Tensor>(8);

  const TensorShape& embeddings_shape = embeddings->Shape();
  const TensorShape& hidden_shape = hidden_states->Shape();
  ORT_RETURN_IF_NOT(embeddings_shape.NumDimensions() == 3,
                    "embeddings must have shape (batch_size, sequence_length, embedding_size)");
  ORT_RETURN_IF_NOT(hidden_shape.NumDimensions() == 4,
                    "hidden_states must have shape (batch_size, sequence_length, hc_mult, hidden_size)");
  const int64_t batch_size = hidden_shape[0];
  const int64_t sequence_length = hidden_shape[1];
  const int64_t hc_mult = hidden_shape[2];
  const int64_t hidden_size = hidden_shape[3];
  const int64_t embedding_size = embeddings_shape[2];
  ORT_RETURN_IF_NOT(embeddings_shape[0] == batch_size && embeddings_shape[1] == sequence_length,
                    "embeddings and hidden_states batch/sequence dimensions must match");
  ORT_RETURN_IF_NOT(key_weight->Shape() == TensorShape({hc_mult, embedding_size, hidden_size}),
                    "key_weight must have shape (hc_mult, embedding_size, hidden_size)");
  ORT_RETURN_IF_NOT(value_weight->Shape() == TensorShape({embedding_size, hidden_size}),
                    "value_weight must have shape (embedding_size, hidden_size)");
  ORT_RETURN_IF_NOT(key_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "key_norm_scale must have shape (hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(query_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "query_norm_scale must have shape (hc_mult, hidden_size)");
  if (key_bias != nullptr) {
    ORT_RETURN_IF_NOT(key_bias->Shape() == TensorShape({hc_mult, hidden_size}),
                      "key_bias must have shape (hc_mult, hidden_size)");
  }
  if (value_bias != nullptr) {
    ORT_RETURN_IF_NOT(value_bias->Shape() == TensorShape({hidden_size}),
                      "value_bias must have shape (hidden_size)");
  }
  if (conv_norm_scale != nullptr) {
    ORT_RETURN_IF_NOT(conv_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                      "conv_norm_scale must have shape (hc_mult, hidden_size)");
  }

  Tensor* output = context->Output(0, hidden_shape);
  Tensor* output_normed = context->OutputCount() > 1 ? context->Output(1, hidden_shape) : nullptr;
  ORT_RETURN_IF_NOT(output_normed == nullptr || conv_norm_scale != nullptr,
                    "conv_norm_scale is required to produce the gated_value_normed output");
  if (hidden_shape.Size() == 0) {
    return Status::OK();
  }

  const T* embeddings_data = embeddings->Data<T>();
  const T* hidden_data = hidden_states->Data<T>();
  const T* key_weight_data = key_weight->Data<T>();
  const T* key_bias_data = key_bias == nullptr ? nullptr : key_bias->Data<T>();
  const T* value_weight_data = value_weight->Data<T>();
  const T* value_bias_data = value_bias == nullptr ? nullptr : value_bias->Data<T>();
  const T* key_scale_data = key_norm_scale->Data<T>();
  const T* query_scale_data = query_norm_scale->Data<T>();
  const T* conv_scale_data = conv_norm_scale == nullptr ? nullptr : conv_norm_scale->Data<T>();
  T* output_data = output->MutableData<T>();
  T* output_normed_data = output_normed == nullptr ? nullptr : output_normed->MutableData<T>();

  const int64_t rows = batch_size * sequence_length * hc_mult;
  ThreadPool::TryParallelFor(
      context->GetOperatorThreadPool(), narrow<ptrdiff_t>(rows), static_cast<double>(hidden_size * embedding_size),
      [&](ptrdiff_t begin, ptrdiff_t end) {
        std::vector<float> key(static_cast<size_t>(hidden_size));
        std::vector<float> value(static_cast<size_t>(hidden_size));
        for (int64_t row = begin; row < end; ++row) {
          const int64_t g = row % hc_mult;
          const int64_t token = row / hc_mult;
          const T* embedding_row = embeddings_data + token * embedding_size;
          const T* hidden_row = hidden_data + row * hidden_size;

          float key_sum_sq = 0.0f;
          for (int64_t c = 0; c < hidden_size; ++c) {
            float projection = key_bias_data == nullptr ? 0.0f : static_cast<float>(key_bias_data[g * hidden_size + c]);
            for (int64_t e = 0; e < embedding_size; ++e) {
              projection += static_cast<float>(embedding_row[e]) *
                            static_cast<float>(key_weight_data[(g * embedding_size + e) * hidden_size + c]);
            }
            key[static_cast<size_t>(c)] = projection;
            key_sum_sq += projection * projection;

            float value_projection = value_bias_data == nullptr ? 0.0f : static_cast<float>(value_bias_data[c]);
            for (int64_t e = 0; e < embedding_size; ++e) {
              value_projection += static_cast<float>(embedding_row[e]) *
                                  static_cast<float>(value_weight_data[e * hidden_size + c]);
            }
            value[static_cast<size_t>(c)] = value_projection;
          }

          float query_sum_sq = 0.0f;
          for (int64_t c = 0; c < hidden_size; ++c) {
            const float query_value = static_cast<float>(hidden_row[c]);
            query_sum_sq += query_value * query_value;
          }

          const float key_inv_rms = 1.0f / std::sqrt(key_sum_sq / static_cast<float>(hidden_size) + epsilon_);
          const float query_inv_rms = 1.0f / std::sqrt(query_sum_sq / static_cast<float>(hidden_size) + epsilon_);
          float dot = 0.0f;
          for (int64_t c = 0; c < hidden_size; ++c) {
            const float normed_key = key[static_cast<size_t>(c)] * key_inv_rms *
                                     static_cast<float>(key_scale_data[g * hidden_size + c]);
            const float normed_query = static_cast<float>(hidden_row[c]) * query_inv_rms *
                                       static_cast<float>(query_scale_data[g * hidden_size + c]);
            dot += normed_key * normed_query;
          }
          dot /= std::sqrt(static_cast<float>(hidden_size));
          const float gate_arg = kernel_helper::EngramGateArg(dot);
          const float gate = kernel_helper::SigmoidFloat(gate_arg);

          T* output_row = output_data + row * hidden_size;
          float gated_sum_sq = 0.0f;
          for (int64_t c = 0; c < hidden_size; ++c) {
            const float gated_value = gate * value[static_cast<size_t>(c)];
            gated_sum_sq += gated_value * gated_value;
            output_row[c] = static_cast<T>(gated_value);
          }

          if (output_normed_data != nullptr) {
            const float normed_inv_rms =
                1.0f / std::sqrt(gated_sum_sq / static_cast<float>(hidden_size) + epsilon_);
            T* output_normed_row = output_normed_data + row * hidden_size;
            for (int64_t c = 0; c < hidden_size; ++c) {
              output_normed_row[c] = static_cast<T>(static_cast<float>(output_row[c]) * normed_inv_rms *
                                                    static_cast<float>(conv_scale_data[g * hidden_size + c]));
            }
          }
        }
      });

  return Status::OK();
}

template class EngramGate<float>;
template class EngramGate<MLFloat16>;

}  // namespace contrib
}  // namespace onnxruntime
