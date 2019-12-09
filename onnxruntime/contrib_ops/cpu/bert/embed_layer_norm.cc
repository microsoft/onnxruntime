// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm.h"
#include "embed_layer_norm_helper.h"
#include "core/util/math_cpuonly.h"
#include "core/platform/threadpool.h"

#include <atomic>

namespace onnxruntime {
namespace contrib {
// These ops are internal-only, so register outside of onnx
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      EmbedLayerNormalization,                                    \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      EmbedLayerNorm<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
EmbedLayerNorm<T>::EmbedLayerNorm(const OpKernelInfo& info) : OpKernel(info) {}

template <typename T>
Status EmbedLayerNorm<T>::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(embed_layer_norm::CheckInputs(context));

  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* segment_ids = context->Input<Tensor>(1);
  const Tensor* word_embedding = context->Input<Tensor>(2);
  const Tensor* position_embedding = context->Input<Tensor>(3);
  const Tensor* segment_embedding = context->Input<Tensor>(4);
  const Tensor* gamma = context->Input<Tensor>(5);
  const Tensor* beta = context->Input<Tensor>(6);
  const Tensor* mask = context->Input<Tensor>(7);  // optional. nullptr if not provided

  const auto input_dims = input_ids->Shape().GetDims();
  int64_t hidden_size = word_embedding->Shape()[1];

  std::vector<int64_t> out_dims;
  out_dims.reserve(3);
  out_dims.push_back(input_dims[0]);
  out_dims.push_back(input_dims[1]);
  out_dims.push_back(hidden_size);
  TensorShape output_shape(out_dims);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> mask_index_dims;
  mask_index_dims.push_back(input_dims[0]);
  TensorShape mask_index_shape(mask_index_dims);
  Tensor* mask_index = context->Output(1, mask_index_shape);

  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length = static_cast<int>(input_dims[1]);

  int word_embedding_length = static_cast<int>(word_embedding->Shape()[0]);
  int position_embedding_length = static_cast<int>(position_embedding->Shape()[0]);
  int segment_embedding_length = static_cast<int>(segment_embedding->Shape()[0]);

  auto input_ids_data = input_ids->template Data<int32_t>();
  auto segment_ids_data = segment_ids->template Data<int32_t>();
  auto word_embedding_data = word_embedding->template Data<T>();
  auto position_embedding_data = position_embedding->template Data<T>();
  auto segment_embedding_data = segment_embedding->template Data<T>();
  auto gamma_data = gamma->template Data<T>();
  auto beta_data = beta->template Data<T>();
  auto output_data = output->template MutableData<T>();

  // Calculate output
  {
    std::atomic_bool failed{false};

    int n = batch_size * sequence_length;
    concurrency::ThreadPool::TryBatchParallelFor(context->GetOperatorThreadPool(), n, [=, &failed](int index) {
      int word_col_index = input_ids_data[index];
      if (word_col_index < 0 || word_col_index >= word_embedding_length) {
        failed.store(true, std::memory_order_release);
        return;
      }
      int position_col_index = index % sequence_length;
      if (position_col_index >= position_embedding_length) {
        failed.store(true, std::memory_order_release);
        return;
      }
      int segment_col_index = segment_ids_data[index];
      if (segment_col_index < 0 || segment_col_index >= segment_embedding_length) {
        failed.store(true, std::memory_order_release);
        return;
      }

      T* y = output_data + index * hidden_size;
      const T* input_word_embedding = word_embedding_data + word_col_index * hidden_size;
      const T* input_position_embedding = position_embedding_data + position_col_index * hidden_size;
      const T* input_segment_embedding = segment_embedding_data + segment_col_index * hidden_size;

      T sum = static_cast<T>(0);
      for (int i = 0; i < hidden_size; i++) {
        T subtotal = input_word_embedding[i] + input_position_embedding[i] + input_segment_embedding[i];
        y[i] = subtotal;
        sum += subtotal;
      }
      T mean = sum / hidden_size;
      sum = 0;
      for (int i = 0; i < hidden_size; i++) {
        T a = y[i] - mean;
        y[i] = a;
        sum += a * a;
      }
      T e = sqrt(sum / hidden_size + static_cast<T>(1.0e-13));
      for (int i = 0; i < hidden_size; i++) {
        y[i] = y[i] / e * gamma_data[i] + beta_data[i];
      }
    });

    if (failed.load(std::memory_order_acquire)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input index out of range");
    }
  }

  // Calculate mask
  if (nullptr != mask) {
    const int32_t* mask_data = mask->template Data<int32_t>();
    for (int b = 0; b < batch_size; b++) {
      mask_index->template MutableData<int32_t>()[b] = static_cast<int32_t>(std::count_if(mask_data + (b * sequence_length),
                                                                                          mask_data + (b * sequence_length) + sequence_length,
                                                                                          [](int v) { return v == 1; }));
    }
  } else {
    memset(mask_index->template MutableData<int32_t>(), 0, batch_size * sizeof(int32_t));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
