// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/defs/tensor_proto_util.h"
#include "embed_layer_norm.h"
#include "embed_layer_norm_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      EmbedLayerNormalization,                                    \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      EmbedLayerNorm<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

using namespace ONNX_NAMESPACE;

template <typename T>
EmbedLayerNorm<T>::EmbedLayerNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
}

template <typename T>
Status EmbedLayerNorm<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* segment_ids = context->Input<Tensor>(1);
  const Tensor* mask = context->Input<Tensor>(2);
  const Tensor* word_embedding = context->Input<Tensor>(3);
  const Tensor* position_embedding = context->Input<Tensor>(4);
  const Tensor* segment_embedding = context->Input<Tensor>(5);
  const Tensor* gamma = context->Input<Tensor>(6);
  const Tensor* beta = context->Input<Tensor>(7);

  if (input_ids->Shape() != segment_ids->Shape() || input_ids->Shape() != mask->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0, 1 and 2 shall have same shape");
  }

  const auto input_dims = input_ids->Shape().GetDims();
  if (input_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input_ids is expected to have 2 dimensions, got ", input_dims.size());
  }

  const auto word_embedding_dims = word_embedding->Shape().GetDims();
  if (word_embedding_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "word_embedding is expected to have 2 dimensions, got ", word_embedding_dims.size());
  }

  const auto position_embedding_dims = position_embedding->Shape().GetDims();
  if (position_embedding_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "position_embedding is expected to have 2 dimensions, got ", position_embedding_dims.size());
  }

  const auto segment_embedding_dims = segment_embedding->Shape().GetDims();
  if (segment_embedding_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "segment_embedding is expected to have 2 dimensions, got ", segment_embedding_dims.size());
  }

  if (word_embedding_dims[1] != position_embedding_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "word_embedding and position_embedding shall have same dimension 1");
  }
  int64_t hidden_size = word_embedding_dims[1];

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
  size_t element_size = sizeof(T);

  if (!LaunchEmbedLayerNormKernel(
          output->template MutableData<T>(),
          mask_index->template MutableData<int32_t>(),
          input_ids->template Data<int32_t>(),
          segment_ids->template Data<int32_t>(),
          mask->template Data<int32_t>(),
          gamma->template Data<T>(),
          beta->template Data<T>(),
          word_embedding->template Data<T>(),
          position_embedding->template Data<T>(),
          segment_embedding->template Data<T>(),
          static_cast<int>(hidden_size),
          batch_size,
          sequence_length,
          element_size)) {
    // Get last error to reset it to cudaSuccess.
    CUDA_CALL(cudaGetLastError());
    return Status(common::ONNXRUNTIME, common::FAIL);
  }

  return Status::OK();
}

}  //namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
