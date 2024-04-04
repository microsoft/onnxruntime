// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/tunable/cuda_tunable.h"
#include "core/providers/cuda/math/gemm.h"

namespace onnxruntime {
namespace cuda {
namespace tunable {

template <typename T>
struct GemmParams : OpParams {
  GemmParams(int m, int n, int k, bool trans_a, bool trans_b, float alpha, float beta, const Gemm<T>* gemm_kernel,
             OpKernelContext* ctx);

  std::string Signature() const override {
    return MakeString((trans_a_ ? "T" : "N"), (trans_b_ ? "T" : "N"), "_", m_, "_", n_, "_", k_, "_", bm_, "_", bn_);
  }

  bool trans_a_;
  bool trans_b_;
  float alpha_;
  float beta_;
  int m_;
  int n_;
  int k_;
  int bm_;
  int bn_;
  const Gemm<T>* gemm_kernel_;
  OpKernelContext* ctx_;
#ifdef ENABLE_TRITON
  bool has_triton_support_ = false;
#endif
};

template <typename T>
common::Status TunableGemm(int m, int n, int k, bool trans_a, bool trans_b, float alpha, float beta,
                           const Gemm<T>* gemm_kernel, OpKernelContext* ctx);

}  // namespace tunable
}  // namespace cuda
}  // namespace onnxruntime
