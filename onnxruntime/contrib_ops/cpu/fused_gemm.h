// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cpu/math/gemm.h"

namespace onnxruntime {
namespace contrib {
template <typename T_X,
          typename T_W,
          typename T_B,
          typename T_Y>
class FusedGemm : public Gemm<T_X, T_W, T_B, T_Y> {
 public:
  FusedGemm(const OpKernelInfo& info) : Gemm<T_X, T_W, T_B, T_Y>(info) {
    Gemm<T_X, T_W, T_B, T_Y>::activation_ = info.GetAttrOrDefault<std::string>("activation", "");
    Gemm<T_X, T_W, T_B, T_Y>::leaky_relu_alpha_ = info.GetAttrOrDefault("leaky_relu_alpha", 0.01f);
  }

  Status Compute(OpKernelContext* context) const override {
    return Gemm<T_X, T_W, T_B, T_Y>::Compute(context);
  }
};
}  // namespace contrib
}  // namespace onnxruntime
