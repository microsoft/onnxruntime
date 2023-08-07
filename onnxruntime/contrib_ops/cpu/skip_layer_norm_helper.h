// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace skip_layer_norm_helper {

template <typename T>
Status CheckInputs(const T* input,
                   const T* skip,
                   const T* gamma,
                   const T* beta,
                   const T* bias,
                   int hidden_size_check,
                   size_t input_dims_size_check) {
  const auto& input_dims_check = input->Shape().GetDims();
  const auto& skip_dims_check = skip->Shape().GetDims();
  size_t skip_dims_size_check = skip_dims_check.size();

  if (skip_dims_size_check != 3 && skip_dims_size_check != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "skip is expected to have 3 or 2 dimensions, got ", skip_dims_size_check);
  }

  if ((input->Shape() != skip->Shape()) && ((skip_dims_check[0] != 1 || skip_dims_size_check != 2) && input_dims_size_check != 3)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "skip is expected to have same shape as input or, a batch size of 1 or no batch size when input has 3 dimensions");
  }

  if (input_dims_size_check != 3 && input_dims_size_check != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input is expected to have 3 or 2 dimensions, got ", input_dims_size_check);
  }

  if (skip_dims_check[skip_dims_size_check - 1] != input_dims_check[input_dims_size_check - 1] || skip_dims_check[skip_dims_size_check - 2] != input_dims_check[input_dims_size_check - 2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "last two dimensions of skip needs to be same as input");
  }

  const auto& gamma_dims = gamma->Shape().GetDims();
  if (gamma_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have 1 dimension, got ", gamma_dims.size());
  }
  if (gamma_dims[0] != hidden_size_check) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Last dimension of gamma and input does not match");
  }

  if (nullptr != beta) {
    const auto& beta_dims = beta->Shape().GetDims();
    if (beta_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "beta is expected to have 1 dimension, got ", beta_dims.size());
    }
    if (beta_dims[0] != hidden_size_check) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Last dimension of beta and input does not match");
    }
  }

  if (nullptr != bias) {
    const auto& bias_dims = bias->Shape().GetDims();
    if (bias_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "bias is expected to have 1 dimension, got ", bias_dims.size());
    }
    if (bias_dims[0] != hidden_size_check) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Last dimension of bias and input does not match");
    }
  }
  return Status::OK();
}

}  // namespace skip_layer_norm_helper
}  // namespace contrib
}  // namespace onnxruntime
