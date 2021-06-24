// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qembed_layer_norm.h"

#include <cmath>

#include "embed_layer_norm_helper.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

namespace {

// TODO(kreeger): Find a global home for this helper method.
template <typename T>
inline float Dequantize(T value, float scale, T zero_point) {
  return static_cast<float>(static_cast<int32_t>(value) - zero_point) * scale;
}

Status CheckQuantizedInputs(OpKernelContext* context) {
  const Tensor* word_embedding_scale_tensor = context->Input<Tensor>(8);
  const Tensor* position_embedding_scale_tensor = context->Input<Tensor>(9);
  const Tensor* segment_embedding_scale_tensor = context->Input<Tensor>(10);
  const Tensor* gamma_scale_tensor = context->Input<Tensor>(11);
  const Tensor* beta_scale_tensor = context->Input<Tensor>(12);
  const Tensor* word_embedding_zero_point_tensor = context->Input<Tensor>(13);
  const Tensor* position_embedding_zero_point_tensor = context->Input<Tensor>(14);
  const Tensor* segment_embedding_zero_point_tensor = context->Input<Tensor>(15);
  const Tensor* gamma_zero_point_tensor = context->Input<Tensor>(16);
  const Tensor* beta_zero_point_tensor = context->Input<Tensor>(17);

  bool has_segment_embedding = context->Input<Tensor>(1) != nullptr;

  if (!IsScalarOr1ElementVector(word_embedding_scale_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Word embedding scale must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(position_embedding_scale_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Position embedding scale must be a scalar or 1D tensor of size 1");
  }

  if (has_segment_embedding && !IsScalarOr1ElementVector(segment_embedding_scale_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Segment embedding scale must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(gamma_scale_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Gamma scale must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(beta_scale_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Beta scale must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(word_embedding_zero_point_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Word embedding zero point must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(position_embedding_zero_point_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Position embedding zero point must be a scalar or 1D tensor of size 1");
  }

  if (has_segment_embedding && !IsScalarOr1ElementVector(segment_embedding_zero_point_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Segment embedding zero point must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(gamma_zero_point_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Gamma zero point must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(beta_zero_point_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Beta zero point must be a scalar or 1D tensor of size 1");
  }

  return Status::OK();
}

}  // namespace

// This op is internal-only, so register outside of onnx:
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      QEmbedLayerNormalization,                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      QEmbedLayerNorm<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
QEmbedLayerNorm<T>::QEmbedLayerNorm(const OpKernelInfo& op_kernel_info)
    : EmbedLayerNormBase(op_kernel_info) {
}

template <typename T>
Status QEmbedLayerNorm<T>::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(embed_layer_norm::CheckInputs(context));
  ORT_RETURN_IF_ERROR(CheckQuantizedInputs(context));

  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* segment_ids = context->Input<Tensor>(1);  // optional. nullptr if it's distill-bert
  const Tensor* word_embedding = context->Input<Tensor>(2);
  const Tensor* position_embedding = context->Input<Tensor>(3);
  const Tensor* segment_embedding = context->Input<Tensor>(4);  // optional. nullptr if it's distill-bert
  const Tensor* gamma = context->Input<Tensor>(5);
  const Tensor* beta = context->Input<Tensor>(6);
  const Tensor* mask = context->Input<Tensor>(7);  // optional. nullptr if not provided
  const Tensor* word_embedding_scale_tensor = context->Input<Tensor>(8);
  const Tensor* position_embedding_scale_tensor = context->Input<Tensor>(9);
  const Tensor* segment_embedding_scale_tensor = context->Input<Tensor>(10);
  const Tensor* gamma_scale_tensor = context->Input<Tensor>(11);
  const Tensor* beta_scale_tensor = context->Input<Tensor>(12);
  const Tensor* word_embedding_zero_point_tensor = context->Input<Tensor>(13);
  const Tensor* position_embedding_zero_point_tensor = context->Input<Tensor>(14);
  const Tensor* segment_embedding_zero_point_tensor = context->Input<Tensor>(15);
  const Tensor* gamma_zero_point_tensor = context->Input<Tensor>(16);
  const Tensor* beta_zero_point_tensor = context->Input<Tensor>(17);

  bool has_segment_embedding = segment_ids != nullptr;

  const int32_t* input_ids_data = input_ids->template Data<int32_t>();
  const int32_t* segment_ids_data =
      has_segment_embedding ? segment_ids->template Data<int32_t>() : nullptr;

  const auto& input_dims = input_ids->Shape().GetDims();
  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length = static_cast<int>(input_dims[1]);
  int64_t hidden_size = word_embedding->Shape()[1];

  int word_embedding_length = static_cast<int>(word_embedding->Shape()[0]);
  int position_embedding_length = static_cast<int>(position_embedding->Shape()[0]);
  int segment_embedding_length =
      has_segment_embedding ? static_cast<int>(segment_embedding->Shape()[0]) : 0;

  // Grab quantization values:
  float word_embedding_scale = *(word_embedding_scale_tensor->template Data<float>());
  uint8_t word_embedding_zero_point =
      *(word_embedding_zero_point_tensor->template Data<uint8_t>());

  float position_embedding_scale = *(position_embedding_scale_tensor->template Data<float>());
  uint8_t position_embedding_zero_point =
      *(position_embedding_zero_point_tensor->template Data<uint8_t>());

  float segment_embedding_scale =
      has_segment_embedding ? *(segment_embedding_scale_tensor->template Data<float>()) : 0.0f;
  uint8_t segment_embedding_zero_point =
      has_segment_embedding ? *(segment_embedding_zero_point_tensor->template Data<uint8_t>()) : 0;

  float gamma_scale = *(gamma_scale_tensor->template Data<float>());
  uint8_t gamma_zero_point = *(gamma_zero_point_tensor->template Data<uint8_t>());

  float beta_scale = *(beta_scale_tensor->template Data<float>());
  uint8_t beta_zero_point = *(beta_zero_point_tensor->template Data<uint8_t>());

  // Request outputs:
  TensorShape output_shape({batch_size, sequence_length, hidden_size});
  Tensor* output = context->Output(0, output_shape);

  TensorShape mask_index_shape({batch_size});
  Tensor* mask_index = context->Output(1, mask_index_shape);

  // Grab pointers to buffers each Tensor represents:
  const uint8_t* word_embedding_data = word_embedding->template Data<uint8_t>();
  const uint8_t* position_embedding_data = position_embedding->template Data<uint8_t>();
  const uint8_t* segment_embedding_data =
      has_segment_embedding ? segment_embedding->template Data<uint8_t>() : nullptr;
  const uint8_t* gamma_data = gamma->template Data<uint8_t>();
  const uint8_t* beta_data = beta->template Data<uint8_t>();

  T* output_data = output->template MutableData<T>();

  // Perform the Op:
  {
    std::atomic_bool failed{false};

    // TODO: Profile and tune this batch parallel execution based on input size.
    // More info: https://github.com/microsoft/onnxruntime/pull/8124/files#r656629895
    int n = batch_size * sequence_length;
    concurrency::ThreadPool::TryBatchParallelFor(
        context->GetOperatorThreadPool(), n, [=, &failed](ptrdiff_t index) {
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
          int segment_col_index = 0;
          if (nullptr != segment_ids_data) {
            segment_col_index = segment_ids_data[index];
            if (segment_col_index < 0 || segment_col_index >= segment_embedding_length) {
              failed.store(true, std::memory_order_release);
              return;
            }
          }

          // Grab inputs for the embeddings for the current batch index:
          const uint8_t* input_word_embedding = word_embedding_data + word_col_index * hidden_size;
          const uint8_t* input_position_embedding =
              position_embedding_data + position_col_index * hidden_size;
          const uint8_t* input_segment_embedding = nullptr;
          if (segment_embedding_data != nullptr) {
            input_segment_embedding = segment_embedding_data + segment_col_index * hidden_size;
          }

          T* output = output_data + (index * hidden_size);

          T sum = static_cast<T>(0);
          for (int i = 0; i < hidden_size; ++i) {
            T subtotal = Dequantize<uint8_t>(input_word_embedding[i],
                                             word_embedding_scale,
                                             word_embedding_zero_point) +
                         Dequantize<uint8_t>(input_position_embedding[i],
                                             position_embedding_scale,
                                             position_embedding_zero_point);
            if (segment_embedding_data != nullptr) {
              subtotal += Dequantize<uint8_t>(input_segment_embedding[i],
                                              segment_embedding_scale,
                                              segment_embedding_zero_point);
            }
            output[i] = subtotal;
            sum += subtotal;
          }

          T mean = sum / hidden_size;
          sum = 0;

          for (int i = 0; i < hidden_size; i++) {
            T a = output[i] - mean;
            output[i] = a;
            sum += a * a;
          }

          T e = sqrt(sum / hidden_size + epsilon());
          for (int i = 0; i < hidden_size; i++) {
            T cur_gamma = Dequantize<uint8_t>(gamma_data[i], gamma_scale, gamma_zero_point);
            T cur_beta = Dequantize<uint8_t>(beta_data[i], beta_scale, beta_zero_point);
            output[i] = output[i] / e * cur_gamma + cur_beta;
          }
        },
        0);

    if (failed.load(std::memory_order_acquire)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input index out of range");
    }
  }

  // Calculate mask
  if (nullptr != mask) {
    const int32_t* mask_data = mask->template Data<int32_t>();
    int32_t* mask_index_data = mask_index->template MutableData<int32_t>();
    for (int b = 0; b < batch_size; b++) {
      mask_index_data[b] =
          static_cast<int32_t>(std::count_if(mask_data + (static_cast<int64_t>(b) * sequence_length),
                                             mask_data + (static_cast<int64_t>(b) * sequence_length) + sequence_length,
                                             [](int v) { return v == 1; }));
    }
  } else {
    memset(mask_index->template MutableData<int32_t>(), 0, batch_size * sizeof(int32_t));
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
