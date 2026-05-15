// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#endif

namespace onnxruntime {
namespace contrib {
namespace bias_gelu_helper {

#ifndef SHARED_PROVIDER
template <typename TOpKernelContext>
inline Status CheckInputs(const TOpKernelContext* context) {
  const Tensor* input = context->template Input<Tensor>(0);
  const Tensor* bias = context->template Input<Tensor>(1);

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
#else
// SHARED_PROVIDER path: implemented via provider bridge forwarding in
// core/providers/shared_library/provider_bridge_provider.cc,
// dispatched to host implementation through g_host_cpu.bias_gelu_helper__CheckInputs.
Status CheckInputs(const OpKernelContext* context);
#endif

}  // namespace bias_gelu_helper
}  // namespace contrib
}  // namespace onnxruntime
