// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_gelu.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/defs/tensor_proto_util.h"
#include "core/common/safeint.h"
#include "core/framework/tensor.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {
namespace bias_gelu_helper {

Status CheckInputs(const OpKernelContext* context) {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* bias = context->Input<Tensor>(1);

  const auto& input_dims = input->Shape().GetDims();
  if (input_dims.size() < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 is expected to have 1 or more dimensions, got ", input_dims.size());
  }

  if (nullptr != bias) {
    const auto& bias_dims = bias->Shape().GetDims();
    if (bias_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 1 is expected to have 1 dimensions, got ", bias_dims.size());
    }
    if (bias_dims[0] != input_dims[input_dims.size() - 1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 1 dimension 0 should have same length as the last dimension of input 0");
    }
  }

  return Status::OK();
}

}  // namespace bias_gelu_helper
}  // namespace contrib
}  // namespace onnxruntime
