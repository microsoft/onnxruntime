// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm.h"
#include "embed_layer_norm_helper.h"
#include "core/util/math_cpuonly.h"

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
  const Tensor* mask = context->Input<Tensor>(2);
  const Tensor* word_embedding = context->Input<Tensor>(3);
  const Tensor* position_embedding = context->Input<Tensor>(4);
  const Tensor* segment_embedding = context->Input<Tensor>(5);
  const Tensor* gamma = context->Input<Tensor>(6);
  const Tensor* beta = context->Input<Tensor>(7);

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

  ConstEigenArrayMap<T> word_embedding_arr(word_embedding->template Data<T>(), hidden_size, word_embedding_length);
  ConstEigenArrayMap<T> position_embedding_arr(position_embedding->template Data<T>(), hidden_size, position_embedding_length);
  ConstEigenArrayMap<T> segment_embedding_arr(segment_embedding->template Data<T>(), hidden_size, segment_embedding_length);
  ConstEigenVectorMap<T> gamma_vector(gamma->template Data<T>(), hidden_size);
  ConstEigenVectorMap<T> beta_vector(beta->template Data<T>(), hidden_size);
  EigenArrayMap<T> output_arr(output->template MutableData<T>(), hidden_size, batch_size * sequence_length);

  // Calculate output
  {
    size_t index = 0;
    for (int b = 0; b < batch_size; b++) {
      for (int s = 0; s < sequence_length; s++) {
        int word_col_index = input_ids->template Data<int>()[index];
        if (word_col_index < 0 || word_col_index >= word_embedding_length) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "word_col_index out of range");
        }
        int position_col_index = s;
        if (position_col_index >= position_embedding_length) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "position_col_index out of range");
        }
        int segment_col_index = segment_ids->template Data<int>()[index];
        if (segment_col_index < 0 || segment_col_index >= segment_embedding_length) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "segment_col_index out of range");
        }

        output_arr.col(index) = word_embedding_arr.col(word_col_index) +
                                position_embedding_arr.col(position_col_index) +
                                segment_embedding_arr.col(segment_col_index);
        output_arr.col(index) -= output_arr.col(index).mean();
        output_arr.col(index) /= static_cast<T>(sqrt(output_arr.col(index).pow(2).mean() + 1.0e-13));
        output_arr.col(index) *= gamma_vector.array();
        output_arr.col(index) += beta_vector.array();
        index++;
      }
    }
  }

  // Calculate mask
  {
    const int* mask_data = mask->template Data<int>();
    for (int b = 0; b < batch_size; b++) {
      mask_index->template MutableData<int>()[b] = static_cast<int>(std::count_if(mask_data + (b * sequence_length),
                                                                                  mask_data + (b * sequence_length) + sequence_length,
                                                                                  [](int v) { return v == 1; }));
    }
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
