// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {

inline Status PrepareGemm29(const TensorShape& a, const TensorShape& b, const TensorShape* c,
                            bool trans_a, bool trans_b, MatMulComputeHelper& helper) {
  ORT_RETURN_IF_NOT(a.NumDimensions() >= 1 && b.NumDimensions() == 2,
                    "Gemm 29 requires A with rank >= 1 and B with rank 2; got A=", a, " and B=", b,
                    ". Reshape the inputs to the required ranks.");
  const auto status = helper.Compute(a, b, trans_a && a.NumDimensions() > 1, trans_b);
  ORT_RETURN_IF_NOT(status.IsOK(), "Gemm 29 A=", a, ", B=", b, ": ", status.ErrorMessage(),
                    ". Match the contraction dimension K after applying transA and transB.");
  if (c) {
    const auto& y = helper.OutputShape();
    ORT_RETURN_IF_NOT(c->NumDimensions() <= y.NumDimensions(),
                      "Gemm 29 bias C=", *c, " cannot broadcast to Y=", y, ". Remove excess leading dimensions.");
    const size_t offset = y.NumDimensions() - c->NumDimensions();
    for (size_t i = 0; i < c->NumDimensions(); ++i) {
      ORT_RETURN_IF_NOT((*c)[i] == 1 || (*c)[i] == y[offset + i],
                        "Gemm 29 bias C=", *c, " cannot broadcast to Y=", y,
                        ". Each aligned bias dimension must be 1 or match the output.");
    }
  }
  return Status::OK();
}

struct GroupedMatMulShape {
  int64_t m{}, k{}, g{}, n{}, s{};

  Status Compute(const TensorShape& input, const TensorShape& weights, const TensorShape& indices,
                 const TensorShape* bias, bool trans_b) {
    ORT_RETURN_IF_NOT(input.NumDimensions() == 2 && weights.NumDimensions() == 3 && indices.NumDimensions() == 2,
                      "GroupedMatMul requires input [M,K], weights [G,K,N] (or [G,N,K] with transB), "
                      "and group_indices [M,S]; got ",
                      input, ", ", weights, ", ", indices,
                      ". Reshape the inputs to these ranks.");
    m = input[0];
    k = input[1];
    g = weights[0];
    n = weights[trans_b ? 1 : 2];
    s = indices[1];
    ORT_RETURN_IF_NOT(weights[trans_b ? 2 : 1] == k && indices[0] == m,
                      "GroupedMatMul input=", input, ", weights=", weights, ", group_indices=", indices,
                      ": match the contraction dimension K and the token dimension M.");
    ORT_RETURN_IF_NOT(!bias || (bias->NumDimensions() == 2 && (*bias)[0] == g && (*bias)[1] == n),
                      "GroupedMatMul bias must have shape [", g, ",", n, "]. Reshape the per-group bias.");
    return Status::OK();
  }
};

}  // namespace onnxruntime
