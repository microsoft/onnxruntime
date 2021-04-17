// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm_helper.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/defs/tensor_proto_util.h"

namespace onnxruntime {
namespace contrib {
void Link_embed_layer_norm_helper() {}
namespace embed_layer_norm {

Status CheckInputs(const OpKernelContext* context) {
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* segment_ids = context->Input<Tensor>(1);  // optional. nullptr if it's distill-bert
  const Tensor* word_embedding = context->Input<Tensor>(2);
  const Tensor* position_embedding = context->Input<Tensor>(3);
  const Tensor* segment_embedding = context->Input<Tensor>(4);  // optional. nullptr if it's distill-bert
  const Tensor* gamma = context->Input<Tensor>(5);
  const Tensor* beta = context->Input<Tensor>(6);
  const Tensor* mask = context->Input<Tensor>(7);  // optional. nullptr if not provided

  if (nullptr != segment_ids && input_ids->Shape() != segment_ids->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 and 1 shall have same shape");
  }

  if (nullptr != mask && input_ids->Shape() != mask->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 and 7 (mask) shall have same shape");
  }

  const auto& input_dims = input_ids->Shape().GetDims();
  if (input_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input_ids is expected to have 2 dimensions, got ", input_dims.size());
  }

  const auto& word_embedding_dims = word_embedding->Shape().GetDims();
  if (word_embedding_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "word_embedding is expected to have 2 dimensions, got ", word_embedding_dims.size());
  }

  const auto& position_embedding_dims = position_embedding->Shape().GetDims();
  if (position_embedding_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "position_embedding is expected to have 2 dimensions, got ", position_embedding_dims.size());
  }

  if (nullptr != segment_embedding) {
    const auto& segment_embedding_dims = segment_embedding->Shape().GetDims();
    if (segment_embedding_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "segment_embedding is expected to have 2 dimensions, got ", segment_embedding_dims.size());
    }
    if (word_embedding_dims[1] != segment_embedding_dims[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "word_embedding and segment_embedding shall have same dimension 1");
    }
  }

  if (word_embedding_dims[1] != position_embedding_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "word_embedding and position_embedding shall have same dimension 1");
  }

  const auto& beta_dims = beta->Shape().GetDims();
  if (beta_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "beta is expected to have 1 dimensions, got ", beta_dims.size());
  }

  if (beta_dims[0] != word_embedding_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "beta is expected to have size of ", word_embedding_dims[1], ", got ", beta_dims[0]);
  }

  const auto& gamma_dims = gamma->Shape().GetDims();
  if (gamma_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have 1 dimensions, got ", gamma_dims.size());
  }

  if (gamma_dims[0] != word_embedding_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have size of ", word_embedding_dims[1], ", got ", gamma_dims[0]);
  }

  return Status::OK();
}

}  // namespace embed_layer_norm
}  // namespace contrib
}  // namespace onnxruntime
