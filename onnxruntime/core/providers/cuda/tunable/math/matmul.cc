// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tunable/math/matmul.h"

#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace cuda {
namespace tunable {

template <typename T>
MatMulParams<T>::MatMulParams(float alpha, bool trans_a, bool trans_b, bool trans_batch_a, bool trans_batch_b,
                              MatMulComputeHelper& helper, const MatMul<T>* matmul_kernel, OpKernelContext* ctx)
    : OpParams(matmul_kernel->GetTuningContext(), ctx->GetComputeStream()),
      alpha_(alpha),
      trans_a_(trans_a),
      trans_b_(trans_b),
      trans_batch_a_(trans_batch_a),
      trans_batch_b_(trans_batch_b),
      helper_(helper),
      matmul_kernel_(matmul_kernel),
      ctx_(ctx) {
}

template <typename T>
std::string MatMulParams<T>::Signature() const {
  const TensorShape& shape_x = ctx_->Input<Tensor>(0)->Shape();
  const TensorShape& shape_y = ctx_->Input<Tensor>(1)->Shape();
  return MakeString((trans_a_ ? "T" : "N"), (trans_b_ ? "T" : "N"), (trans_batch_a_ ? "T" : "N"),
                    (trans_batch_a_ ? "T" : "N"), "_", shape_x.ToString(), "_", shape_y.ToString());
}

namespace {

template <typename T>
common::Status DefaultMatMulOp(const MatMulParams<T>* params) {
  return params->matmul_kernel_->ComputeDefault(params->ctx_, params->helper_);
}

template <typename T>
class MatMulTunableOp : public TunableOp<MatMulParams<T>> {
 public:
  MatMulTunableOp() {
    this->RegisterOp(DefaultMatMulOp<T>);
  }
};

}  // namespace

template <typename T>
inline common::Status TunableMatMul(float alpha, bool trans_a, bool trans_b, bool trans_batch_a, bool trans_batch_b,
                                    MatMulComputeHelper& helper, const MatMul<T>* matmul_kernel, OpKernelContext* ctx) {
  MatMulParams<T> params(alpha, trans_a, trans_b, trans_batch_a, trans_batch_b, helper, matmul_kernel, ctx);
  if (params.tuning_ctx->IsTunableOpEnabled()) {
    static MatMulTunableOp<T> matmul{};
    return matmul(&params);
  }

  return DefaultMatMulOp(&params);
}

#define SPECIALIZE_TUNABLE_MATMUL(T)                                                                    \
  template common::Status TunableMatMul<T>(float alpha, bool trans_a, bool trans_b, bool trans_batch_a, \
                                           bool trans_batch_b, MatMulComputeHelper& helper,             \
                                           const MatMul<T>* matmul_kernel, OpKernelContext* ctx);

SPECIALIZE_TUNABLE_MATMUL(float)
SPECIALIZE_TUNABLE_MATMUL(double)
SPECIALIZE_TUNABLE_MATMUL(MLFloat16)
SPECIALIZE_TUNABLE_MATMUL(BFloat16)

#undef SPECIALIZE_TUNABLE_MATMUL

}  // namespace tunable
}  // namespace cuda
}  // namespace onnxruntime
