// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/engram_gate.h"

#include <cmath>

#include "contrib_ops/cpu/bert/engram_helper.h"
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
  const Tensor* key = context->Input<Tensor>(0);
  const Tensor* query = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* key_norm_scale = context->Input<Tensor>(3);
  const Tensor* query_norm_scale = context->Input<Tensor>(4);

  const TensorShape& key_shape = key->Shape();
  ORT_RETURN_IF_NOT(key_shape.NumDimensions() == 4,
                    "key must have shape (batch_size, sequence_length, hc_mult, hidden_size)");
  const int64_t batch_size = key_shape[0];
  const int64_t sequence_length = key_shape[1];
  const int64_t hc_mult = key_shape[2];
  const int64_t hidden_size = key_shape[3];

  ORT_RETURN_IF_NOT(query->Shape() == key_shape, "query must have the same shape as key");
  ORT_RETURN_IF_NOT(value->Shape() == TensorShape({batch_size, sequence_length, hidden_size}),
                    "value must have shape (batch_size, sequence_length, hidden_size)");
  ORT_RETURN_IF_NOT(key_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "key_norm_scale must have shape (hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(query_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "query_norm_scale must have shape (hc_mult, hidden_size)");

  Tensor* output = context->Output(0, key_shape);
  if (key_shape.Size() == 0) {
    return Status::OK();
  }

  const T* key_data = key->Data<T>();
  const T* query_data = query->Data<T>();
  const T* value_data = value->Data<T>();
  const T* key_scale_data = key_norm_scale->Data<T>();
  const T* query_scale_data = query_norm_scale->Data<T>();
  T* output_data = output->MutableData<T>();

  const int64_t rows = batch_size * sequence_length * hc_mult;
  ThreadPool::TryParallelFor(
      // Each row makes one fused reduction pass and one output pass over hidden_size, plus a
      // handful of scalar transcendentals. Costing it as a single pass would over-partition.
      context->GetOperatorThreadPool(), narrow<ptrdiff_t>(rows),
      static_cast<double>(2 * hidden_size + 32),
      [&](ptrdiff_t begin, ptrdiff_t end) {
        for (int64_t row = begin; row < end; ++row) {
          const int64_t g = row % hc_mult;
          const int64_t token = row / hc_mult;
          const T* key_row = key_data + row * hidden_size;
          const T* query_row = query_data + row * hidden_size;
          const T* value_row = value_data + token * hidden_size;

          // Both inverse RMS factors are scalars, so they can be pulled out of the dot product and
          // applied afterwards. That folds the two reductions into one pass over key_row and
          // query_row, which is what the CUDA and WGSL kernels already do.
          const T* key_scale_row = key_scale_data + g * hidden_size;
          const T* query_scale_row = query_scale_data + g * hidden_size;
          float key_sum_sq = 0.0f;
          float query_sum_sq = 0.0f;
          float dot_numerator = 0.0f;
          for (int64_t c = 0; c < hidden_size; ++c) {
            const float key_value = static_cast<float>(key_row[c]);
            const float query_value = static_cast<float>(query_row[c]);
            key_sum_sq += key_value * key_value;
            query_sum_sq += query_value * query_value;
            dot_numerator += key_value * static_cast<float>(key_scale_row[c]) *
                             query_value * static_cast<float>(query_scale_row[c]);
          }

          const float key_inv_rms = 1.0f / std::sqrt(key_sum_sq / static_cast<float>(hidden_size) + epsilon_);
          const float query_inv_rms = 1.0f / std::sqrt(query_sum_sq / static_cast<float>(hidden_size) + epsilon_);
          const float dot =
              dot_numerator * key_inv_rms * query_inv_rms / std::sqrt(static_cast<float>(hidden_size));
          const float gate = engram_helper::SigmoidFloat(engram_helper::EngramGateArg(dot));

          T* output_row = output_data + row * hidden_size;
          for (int64_t c = 0; c < hidden_size; ++c) {
            output_row[c] = static_cast<T>(gate * static_cast<float>(value_row[c]));
          }
        }
      });

  return Status::OK();
}

template class EngramGate<float>;
template class EngramGate<MLFloat16>;

}  // namespace contrib
}  // namespace onnxruntime
