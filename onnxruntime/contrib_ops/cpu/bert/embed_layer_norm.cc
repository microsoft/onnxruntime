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

EmbedLayerNormBase::EmbedLayerNormBase(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
  ORT_ENFORCE(epsilon_ >= 0);
}

float EmbedLayerNormBase::epsilon() const {
  return epsilon_;
}

template <typename T>
EmbedLayerNorm<T>::EmbedLayerNorm(const OpKernelInfo& op_kernel_info)
    : EmbedLayerNormBase(op_kernel_info) {
}

template <typename T>
Status EmbedLayerNorm<T>::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(embed_layer_norm::CheckInputs(context));
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* segment_ids = context->Input<Tensor>(1);  // optional. nullptr if it's distill-bert
  const Tensor* word_embedding = context->Input<Tensor>(2);
  const Tensor* position_embedding = context->Input<Tensor>(3);
  const Tensor* segment_embedding = context->Input<Tensor>(4);  // optional. nullptr if it's distill-bert
  const Tensor* gamma = context->Input<Tensor>(5);
  const Tensor* beta = context->Input<Tensor>(6);
  const Tensor* mask = context->Input<Tensor>(7);          // optional. nullptr if not provided
  const Tensor* position_ids = context->Input<Tensor>(8);  // optional. nullptr if not provided

  const auto& input_dims = input_ids->Shape().GetDims();
  int64_t hidden_size = word_embedding->Shape()[1];

  TensorShape output_shape({input_dims[0], input_dims[1], hidden_size});
  Tensor* output = context->Output(0, output_shape);

  TensorShape mask_index_shape({input_dims[0]});
  Tensor* mask_index = context->Output(1, mask_index_shape);

  Tensor* embedding_sum = context->Output(2, output_shape);

  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length = static_cast<int>(input_dims[1]);

  int word_embedding_length = static_cast<int>(word_embedding->Shape()[0]);
  int position_embedding_length = static_cast<int>(position_embedding->Shape()[0]);
  int segment_embedding_length = (nullptr == segment_embedding) ? 0 : static_cast<int>(segment_embedding->Shape()[0]);

  const int32_t* input_ids_data = input_ids->Data<int32_t>();
  const int32_t* segment_ids_data = (nullptr == segment_ids) ? nullptr : segment_ids->Data<int32_t>();
  const T* word_embedding_data = word_embedding->Data<T>();
  const T* position_embedding_data = position_embedding->Data<T>();
  const T* segment_embedding_data = (nullptr == segment_embedding) ? nullptr : segment_embedding->Data<T>();
  const T* gamma_data = gamma->Data<T>();
  const T* beta_data = beta->Data<T>();
  const int32_t* position_ids_data = (nullptr == position_ids) ? nullptr : position_ids->Data<int32_t>();
  const bool broadcast_position_ids = (nullptr != position_ids && position_ids->Shape()[0] == 1) ? true : false;
  T* output_data = output->MutableData<T>();
  T* embedding_sum_data = (embedding_sum != nullptr) ? embedding_sum->MutableData<T>() : nullptr;

  // Calculate output
  {
    std::atomic_bool failed{false};

    int n = batch_size * sequence_length;
    concurrency::ThreadPool::TryBatchParallelFor(
#if __cplusplus >= 202002L
        context->GetOperatorThreadPool(), n, [=, this, &failed](ptrdiff_t index) {
#else
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
#endif
        context->GetOperatorThreadPool(), n, [=, &failed](ptrdiff_t index) {
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif
          int word_col_index = input_ids_data[index];
          if (word_col_index < 0 || word_col_index >= word_embedding_length) {
            failed.store(true, std::memory_order_release);
            return;
          }
          int position_col_index = (position_ids_data == nullptr) ? index % sequence_length : (broadcast_position_ids ? position_ids_data[index % sequence_length] : position_ids_data[index]);
          if (position_col_index >= position_embedding_length) {
            failed.store(true, std::memory_order_release);
            return;
          }
          int segment_col_index = 0;
          if (nullptr != segment_ids_data) {
            segment_col_index = segment_ids_data[index];
            if (segment_col_index < 0 || segment_col_index >= segment_embedding_length) {
              failed.store(true, std::memory_order_release);
              return;
            }
          }

          T* y = output_data + index * hidden_size;
          T* y1 = nullptr;
          if (embedding_sum_data != nullptr) {
            y1 = embedding_sum_data + index * hidden_size;
          }
          const T* input_word_embedding = word_embedding_data + word_col_index * hidden_size;
          const T* input_position_embedding = position_embedding_data + position_col_index * hidden_size;
          const T* input_segment_embedding = (nullptr == segment_embedding_data)
                                                 ? nullptr
                                                 : segment_embedding_data + segment_col_index * hidden_size;

          T sum = static_cast<T>(0);
          for (int i = 0; i < hidden_size; i++) {
            T subtotal = input_word_embedding[i] + input_position_embedding[i];
            if (nullptr != segment_embedding_data)
              subtotal += input_segment_embedding[i];
            y[i] = subtotal;
            if (y1 != nullptr) {
              y1[i] = subtotal;
            }
            sum += subtotal;
          }
          T mean = sum / hidden_size;
          sum = 0;
          for (int i = 0; i < hidden_size; i++) {
            T a = y[i] - mean;
            y[i] = a;
            sum += a * a;
          }
          T e = sqrt(sum / hidden_size + static_cast<T>(epsilon()));
          for (int i = 0; i < hidden_size; i++) {
            y[i] = y[i] / e * gamma_data[i] + beta_data[i];
          }
        },
        0);

    if (failed.load(std::memory_order_acquire)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input index out of range");
    }
  }

  // Calculate mask
  if (nullptr != mask && nullptr != mask_index) {
    const int32_t* mask_data = mask->Data<int32_t>();
    int32_t* mask_index_data = mask_index->MutableData<int32_t>();
    for (int b = 0; b < batch_size; b++) {
      int32_t cur_sum = 0;
      const int32_t* cur_mask_data = mask_data + (static_cast<int64_t>(b) * sequence_length);
      for (int s = 0; s < sequence_length; ++s) {
        if (cur_mask_data[s] == 1) {
          cur_sum += cur_mask_data[s];
        }
      }
      mask_index_data[b] = cur_sum;
    }
  } else if (mask_index != nullptr) {
    memset(mask_index->MutableData<int32_t>(), 0, batch_size * sizeof(int32_t));
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
