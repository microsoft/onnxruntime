// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/engram_ops.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "core/common/narrow.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

#define REGISTER_SHORT_CONV_TYPED(T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ShortConv,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ShortConv<T>);

REGISTER_SHORT_CONV_TYPED(float)

#undef REGISTER_SHORT_CONV_TYPED

#define REGISTER_NGRAM_HASH_TYPED(T)                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      NgramHashMapping,                                           \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<T>()), \
      NgramHashMapping<T>);

REGISTER_NGRAM_HASH_TYPED(int32_t)
REGISTER_NGRAM_HASH_TYPED(int64_t)

#undef REGISTER_NGRAM_HASH_TYPED

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

#undef REGISTER_ENGRAM_GATE_TYPED

namespace {

inline float SigmoidFloat(float x) {
  if (x > 0.0f) {
    return 1.0f / (1.0f + std::exp(-x));
  }
  const float exp_x = std::exp(x);
  return exp_x / (1.0f + exp_x);
}

inline float SiluFloat(float x) {
  return x * SigmoidFloat(x);
}

template <typename T>
T PositiveMod(T value, T mod) {
  T result = value % mod;
  if (result < 0) {
    result += mod;
  }
  return result;
}

template <typename T>
T WrappedMultiply(T a, T b) {
  using UnsignedT = typename std::make_unsigned<T>::type;
  return static_cast<T>(static_cast<UnsignedT>(a) * static_cast<UnsignedT>(b));
}

}  // namespace

template <typename T>
ShortConv<T>::ShortConv(const OpKernelInfo& info) : OpKernel(info) {
  activation_ = info.GetAttrOrDefault<std::string>("activation", "silu");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");
  dilation_ = info.GetAttrOrDefault<int64_t>("dilation", 1);
  ORT_ENFORCE(dilation_ >= 1, "dilation must be >= 1");
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

template <typename T>
Status ShortConv<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weight = context->Input<Tensor>(1);
  const Tensor* norm_scale = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);

  const TensorShape& input_shape = input->Shape();
  const TensorShape& weight_shape = weight->Shape();
  const TensorShape& scale_shape = norm_scale->Shape();

  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 4,
                    "input must have shape (batch_size, sequence_length, hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 3,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  ORT_RETURN_IF_NOT(scale_shape.NumDimensions() == 2,
                    "norm_scale must have shape (hc_mult, hidden_size)");

  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  const int64_t hc_mult = input_shape[2];
  const int64_t hidden_size = input_shape[3];
  const int64_t channels = hc_mult * hidden_size;
  const int64_t kernel_size = weight_shape[2];

  ORT_RETURN_IF_NOT(scale_shape[0] == hc_mult && scale_shape[1] == hidden_size,
                    "norm_scale shape must match input hc_mult and hidden_size");
  ORT_RETURN_IF_NOT(weight_shape[0] == channels && weight_shape[1] == 1,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  if (bias != nullptr) {
    ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 1 && bias->Shape()[0] == channels,
                      "bias must have shape (hc_mult * hidden_size)");
  }

  Tensor* output = context->Output(0, input_shape);
  if (input_shape.Size() == 0) {
    return Status::OK();
  }

  const T* input_data = input->Data<T>();
  const T* weight_data = weight->Data<T>();
  const T* scale_data = norm_scale->Data<T>();
  const T* bias_data = bias == nullptr ? nullptr : bias->Data<T>();
  T* output_data = output->MutableData<T>();
  const bool apply_silu = activation_ == "silu" || activation_ == "swish";
  const int64_t total = batch_size * sequence_length * channels;

  ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), narrow<ptrdiff_t>(total),
      [&](ptrdiff_t begin, ptrdiff_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
          const int64_t c = linear % hidden_size;
          const int64_t g = (linear / hidden_size) % hc_mult;
          const int64_t t = (linear / channels) % sequence_length;
          const int64_t b = linear / (sequence_length * channels);
          const int64_t flat_channel = g * hidden_size + c;

          float sum = bias_data == nullptr ? 0.0f : static_cast<float>(bias_data[flat_channel]);
          for (int64_t k = 0; k < kernel_size; ++k) {
            const int64_t source_t = t - (kernel_size - 1 - k) * dilation_;
            if (source_t < 0) {
              continue;
            }

            const int64_t row_base = ((b * sequence_length + source_t) * hc_mult + g) * hidden_size;
            float sum_sq = 0.0f;
            for (int64_t i = 0; i < hidden_size; ++i) {
              const float value = static_cast<float>(input_data[row_base + i]);
              sum_sq += value * value;
            }
            const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_size) + epsilon_);
            const float normed = static_cast<float>(input_data[row_base + c]) * inv_rms *
                                 static_cast<float>(scale_data[g * hidden_size + c]);
            sum += normed * static_cast<float>(weight_data[flat_channel * kernel_size + k]);
          }
          output_data[linear] = static_cast<T>(apply_silu ? SiluFloat(sum) : sum);
        }
      },
      0);

  return Status::OK();
}

template <typename T>
NgramHashMapping<T>::NgramHashMapping(const OpKernelInfo& info) : OpKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("max_ngram_size", &max_ngram_size_).IsOK(),
              "max_ngram_size attribute is required");
  ORT_ENFORCE(info.GetAttr<int64_t>("n_head_per_ngram", &n_head_per_ngram_).IsOK(),
              "n_head_per_ngram attribute is required");
  int64_t pad_id = 0;
  ORT_ENFORCE(info.GetAttr<int64_t>("pad_id", &pad_id).IsOK(), "pad_id attribute is required");
  ORT_ENFORCE(max_ngram_size_ >= 2, "max_ngram_size must be at least 2");
  ORT_ENFORCE(n_head_per_ngram_ >= 1, "n_head_per_ngram must be positive");
  ORT_ENFORCE(pad_id >= static_cast<int64_t>(std::numeric_limits<T>::min()) &&
                  pad_id <= static_cast<int64_t>(std::numeric_limits<T>::max()),
              "pad_id is out of range for the input id type");
  pad_id_ = static_cast<T>(pad_id);
}

template <typename T>
Status NgramHashMapping<T>::Compute(OpKernelContext* context) const {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* multipliers = context->Input<Tensor>(1);
  const Tensor* vocab_sizes = context->Input<Tensor>(2);

  const TensorShape& input_shape = input_ids->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2, "input_ids must have rank 2");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 &&
                        multipliers->Shape()[0] >= max_ngram_size_,
                    "multipliers must have shape (max_ngram_size)");
  const int64_t num_heads = (max_ngram_size_ - 1) * n_head_per_ngram_;
  ORT_RETURN_IF_NOT(vocab_sizes->Shape().NumDimensions() == 1 && vocab_sizes->Shape()[0] == num_heads,
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");

  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  Tensor* output = context->Output(0, TensorShape({batch_size, sequence_length, num_heads}));

  const T* input_data = input_ids->Data<T>();
  const T* multiplier_data = multipliers->Data<T>();
  const T* vocab_data = vocab_sizes->Data<T>();
  T* output_data = output->MutableData<T>();

  const int64_t total = batch_size * sequence_length;
  ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), narrow<ptrdiff_t>(total),
      [&](ptrdiff_t begin, ptrdiff_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
          const int64_t t = linear % sequence_length;
          const int64_t b = linear / sequence_length;
          const int64_t input_base = b * sequence_length;
          const int64_t output_base = linear * num_heads;

          for (int64_t n = 2; n <= max_ngram_size_; ++n) {
            T mix = 0;
            for (int64_t k = 0; k < n; ++k) {
              const int64_t source_t = t - k;
              const T token = source_t < 0 ? pad_id_ : input_data[input_base + source_t];
              const T product = WrappedMultiply(token, multiplier_data[k]);
              mix = k == 0 ? product : static_cast<T>(mix ^ product);
            }

            const int64_t ngram_offset = (n - 2) * n_head_per_ngram_;
            for (int64_t h = 0; h < n_head_per_ngram_; ++h) {
              const int64_t out_h = ngram_offset + h;
              const T mod = vocab_data[out_h];
              output_data[output_base + out_h] = mod <= 0 ? T{} : PositiveMod(mix, mod);
            }
          }
        }
      },
      0);

  return Status::OK();
}

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

  Tensor* output = context->Output(0, hidden_shape);
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
  T* output_data = output->MutableData<T>();

  const int64_t rows = batch_size * sequence_length * hc_mult;
  ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), narrow<ptrdiff_t>(rows),
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
          const float gate_arg = std::copysign(std::sqrt(std::max(std::abs(dot), 1.0e-6f)), dot);
          const float gate = SigmoidFloat(gate_arg);

          T* output_row = output_data + row * hidden_size;
          for (int64_t c = 0; c < hidden_size; ++c) {
            output_row[c] = static_cast<T>(gate * value[static_cast<size_t>(c)]);
          }
        }
      },
      0);

  return Status::OK();
}

template class ShortConv<float>;
template class NgramHashMapping<int32_t>;
template class NgramHashMapping<int64_t>;
template class EngramGate<float>;

}  // namespace contrib
}  // namespace onnxruntime
