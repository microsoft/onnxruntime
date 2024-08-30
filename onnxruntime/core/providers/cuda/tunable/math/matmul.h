// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/tunable/cuda_tunable.h"
#include "core/providers/cuda/math/matmul.h"

namespace onnxruntime {
namespace cuda {
namespace tunable {

template <typename T>
struct MatMulParams : OpParams {
  MatMulParams(float alpha, bool trans_a, bool trans_b, bool trans_batch_a, bool trans_batch_b,
               MatMulComputeHelper& helper, const MatMul<T>* matmul_kernel, OpKernelContext* ctx);

  std::string Signature() const override;

  float alpha_;
  bool trans_a_;
  bool trans_b_;
  bool trans_batch_a_;
  bool trans_batch_b_;
  MatMulComputeHelper& helper_;
  const MatMul<T>* matmul_kernel_;
  OpKernelContext* ctx_;
#ifdef ENABLE_TRITON
  bool has_triton_support_ = false;
#endif
};

template <typename T>
common::Status TunableMatMul(float alpha, bool trans_a, bool trans_b, bool trans_batch_a, bool trans_batch_b,
                             MatMulComputeHelper& helper, const MatMul<T>* matmul_kernel, OpKernelContext* ctx);

}  // namespace tunable
}  // namespace cuda
}  // namespace onnxruntime
