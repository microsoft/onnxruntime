// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>

#include "core/common/safeint.h"
#ifndef SHARED_PROVIDER
#include "core/framework/tensor.h"
#endif

namespace onnxruntime {

struct CausalConvWithStateParameters {
  int64_t batch;
  int64_t channels;
  int64_t length;
  int64_t kernel;
  int64_t history;
  TensorShape state_shape;
};

inline Status PrepareCausalConvWithState(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                         const Tensor* past, int64_t dilation,
                                         CausalConvWithStateParameters& p) {
  const auto& x = input.Shape();
  const auto& w = weight.Shape();
  ORT_RETURN_IF_NOT(x.NumDimensions() == 3 && w.NumDimensions() == 3,
                    "CausalConvWithState requires input [B,C,L] and weight [C,1,K]. Got ",
                    x.ToString(), " and ", w.ToString(), ". Reshape the inputs to rank 3.");
  p.batch = x[0];
  p.channels = x[1];
  p.length = x[2];
  p.kernel = w[2];
  ORT_RETURN_IF_NOT(w[0] == p.channels && w[1] == 1 && p.kernel >= 1,
                    "CausalConvWithState weight must have shape [", p.channels,
                    ",1,K] with K >= 1. Got ", w.ToString(), ". Correct the depthwise weight shape.");
  ORT_RETURN_IF_NOT(dilation >= 1 &&
                        p.kernel - 1 <= (std::numeric_limits<int64_t>::max() - 1) / dilation,
                    "CausalConvWithState requires dilation >= 1 and (K-1)*dilation+1 to fit int64. Got K=",
                    p.kernel, ", dilation=", dilation, ". Reduce the kernel size or dilation.");
  p.history = (p.kernel - 1) * dilation;
  p.state_shape = TensorShape({p.batch, p.channels, p.history});
  ORT_RETURN_IF_NOT(!bias || bias->Shape() == TensorShape({p.channels}),
                    "CausalConvWithState bias must have shape [", p.channels, "]. Correct the bias shape.");
  ORT_RETURN_IF_NOT(!past || past->Shape() == p.state_shape,
                    "CausalConvWithState past_state must have shape ", p.state_shape.ToString(),
                    ". Supply chronological history with (K-1)*dilation samples.");
  return Status::OK();
}

}  // namespace onnxruntime
