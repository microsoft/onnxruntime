// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/float16.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "contrib_ops/rocm/math/gemm_float8_ck.cuh"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;
using namespace onnxruntime::rocm::tunable::blas;

class GemmFloat8 final : public RocmKernel {
 public:
  GemmFloat8(const OpKernelInfo& info) : RocmKernel(info) {
    transA_ = info.GetAttrOrDefault<int64_t>("transA", 0);
    transB_ = info.GetAttrOrDefault<int64_t>("transB", 0);
    dtype_ = info.GetAttrOrDefault<int64_t>("dtype", onnx::TensorProto_DataType_FLOAT16);
    alpha_ = info.GetAttrOrDefault<float>("alpha", 1);
    beta_ = info.GetAttrOrDefault<float>("beta", 0);
  }
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
#if !defined(DISABLE_FLOAT8_TYPES)
  template <typename Fp8T>
  Status ComputeFp8Fp16Fp16(OpKernelContext* ctx, int64_t m, int64_t n, int64_t k,
                            const Tensor* A, const Tensor* scaleA, const Tensor* B, Tensor* C) const;
  template <typename Fp8T>
  Status ComputeFp16Fp8Fp16(OpKernelContext* ctx, int64_t m, int64_t n, int64_t k,
                            const Tensor* A, const Tensor* B, const Tensor* scaleB, Tensor* C) const;

  template <typename TA, typename TB, typename TC, BlasOp OpA, BlasOp OpB>
  [[nodiscard]] inline auto* GetOp() const {
    using OpT = GemmFloat8TunableOp<TA, TB, TC, OpA, OpB>;
    if (tunable_op_) {
      return static_cast<OpT*>(tunable_op_.get());
    }

    auto create = std::make_unique<OpT>();  // avoid new
    tunable_op_ = std::shared_ptr<void>(create.release(), [](void* ptr) {
      auto release = std::unique_ptr<OpT>();  // avoid delete
      release.reset(static_cast<OpT*>(ptr));
    });

    return static_cast<OpT*>(tunable_op_.get());
  }
#endif

  float alpha_;
  float beta_;
  bool transA_;
  bool transB_;
  int64_t dtype_;

  // fully type erased
  mutable std::shared_ptr<void> tunable_op_;
};

Status GemmFloat8::ComputeInternal(OpKernelContext* ctx) const {
#if defined(DISABLE_FLOAT8_TYPES)
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "DISABLE_FLOAT8_TYPES");
#else
  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  const Tensor* C = ctx->Input<Tensor>(2);  // bias
  const Tensor* scale_a = ctx->Input<Tensor>(3);
  const Tensor* scale_b = ctx->Input<Tensor>(4);
  const Tensor* scale_y = ctx->Input<Tensor>(5);

  auto a_shape = A->Shape();
  auto b_shape = B->Shape();
  ORT_ENFORCE(a_shape.NumDimensions() == 2);
  ORT_ENFORCE(b_shape.NumDimensions() == 2);

  auto m = !transA_ ? a_shape[0] : a_shape[1];
  auto k = !transA_ ? a_shape[1] : a_shape[0];
  ORT_ENFORCE(k == (!transB_ ? b_shape[0] : b_shape[1]));  // k is compatiable
  auto n = !transB_ ? b_shape[1] : b_shape[0];

  TensorShapeVector output_shape = {m, n};
  Tensor* Y = ctx->Output(0, output_shape);

  ORT_ENFORCE(!transA_, "ROCm GemmFloat8 does not support input A transpose");
  ORT_ENFORCE(dtype_ == onnx::TensorProto_DataType_FLOAT16, "ROCm GemmFloat8 only supports output float16");
  ORT_ENFORCE(C == nullptr, "ROCm GemmFloat8 does not support bias input");
  ORT_ENFORCE(scale_y == nullptr, "ROCm GemmFloat8 does not support output scaling");

  if (A->IsDataType<Float8E4M3FN>()) {
    return ComputeFp8Fp16Fp16<Float8E4M3FN>(ctx, m, n, k, A, scale_a, B, Y);
  } else if (A->IsDataType<Float8E4M3FNUZ>()) {
    return ComputeFp8Fp16Fp16<Float8E4M3FNUZ>(ctx, m, n, k, A, scale_a, B, Y);
  } else if (B->IsDataType<Float8E4M3FN>()) {
    return ComputeFp16Fp8Fp16<Float8E4M3FN>(ctx, m, n, k, A, B, scale_b, Y);
  } else if (B->IsDataType<Float8E4M3FNUZ>()) {
    return ComputeFp16Fp8Fp16<Float8E4M3FNUZ>(ctx, m, n, k, A, B, scale_b, Y);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unhandled type combination of GemmFloat8");
#endif
}

#if !defined(DISABLE_FLOAT8_TYPES)
template <typename Fp8T>
Status GemmFloat8::ComputeFp8Fp16Fp16(
    OpKernelContext* ctx, int64_t m, int64_t n, int64_t k,
    const Tensor* A, const Tensor* scale_a, const Tensor* B, Tensor* C) const {
  ORT_ENFORCE(A->IsDataType<Fp8T>() && scale_a->IsDataType<float>() && B->IsDataType<MLFloat16>());

  onnxruntime::rocm::tunable::blas::GemmFloat8Params<Fp8T, MLFloat16, MLFloat16> params{};
  params.tuning_ctx = GetTuningContext();
  params.stream = ctx->GetComputeStream();
  params.handle = GetRocblasHandle(ctx);
  params.opa = transA_ ? tunable::blas::BlasOp::Trans : tunable::blas::BlasOp::NonTrans;
  params.opb = transB_ ? tunable::blas::BlasOp::Trans : tunable::blas::BlasOp::NonTrans;

  params.m = m;
  params.n = n;
  params.k = k;

  params.a = static_cast<const Fp8T*>(A->DataRaw());
  params.lda = transA_ ? m : k;
  params.scale_a = alpha_;
  params.scale_a_dev = static_cast<const float*>(scale_a->DataRaw());

  params.b = static_cast<const MLFloat16*>(B->DataRaw());
  params.ldb = transB_ ? k : n;
  params.scale_b = 1.0f;         // NOTE: not used
  params.scale_b_dev = nullptr;  // NOTE: not used

  params.c = static_cast<MLFloat16*>(C->MutableDataRaw());
  params.ldc = n;
  params.scale_c = 1.0f;         // NOTE: not implemented
  params.scale_c_dev = nullptr;  // NOTE: not implemented

  if (!transA_ && !transB_) {
    return (*GetOp<Fp8T, MLFloat16, MLFloat16, BlasOp::NonTrans, BlasOp::NonTrans>())(&params);
  } else if (transA_ && !transB_) {
    ORT_NOT_IMPLEMENTED("transA is not implemented");
  } else if (!transA_ && transB_) {
    ORT_NOT_IMPLEMENTED("transB is not implemented");
  } else if (transA_ && transB_) {
    ORT_NOT_IMPLEMENTED("transA & transB is not implemented");
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unreachable");
}

template <typename Fp8T>
Status GemmFloat8::ComputeFp16Fp8Fp16(
    OpKernelContext* ctx, int64_t m, int64_t n, int64_t k,
    const Tensor* A, const Tensor* B, const Tensor* scale_b, Tensor* C) const {
  ORT_ENFORCE(A->IsDataType<MLFloat16>() && B->IsDataType<Fp8T>() && scale_b->IsDataType<float>());

  onnxruntime::rocm::tunable::blas::GemmFloat8Params<MLFloat16, Fp8T, MLFloat16> params{};
  params.tuning_ctx = GetTuningContext();
  params.stream = ctx->GetComputeStream();
  params.handle = GetRocblasHandle(ctx);
  params.opa = transA_ ? tunable::blas::BlasOp::Trans : tunable::blas::BlasOp::NonTrans;
  params.opb = transB_ ? tunable::blas::BlasOp::Trans : tunable::blas::BlasOp::NonTrans;

  params.m = m;
  params.n = n;
  params.k = k;

  params.a = static_cast<const MLFloat16*>(A->DataRaw());
  params.lda = transA_ ? m : k;
  params.scale_a = 1.0f;         // NOTE: not used
  params.scale_a_dev = nullptr;  // NOTE: not used

  params.b = static_cast<const Fp8T*>(B->DataRaw());
  params.ldb = transB_ ? k : n;
  params.scale_b = alpha_;
  params.scale_b_dev = static_cast<const float*>(scale_b->DataRaw());

  params.c = static_cast<MLFloat16*>(C->MutableDataRaw());
  params.ldc = n;
  params.scale_c = 1.0f;         // NOTE: not implemented
  params.scale_c_dev = nullptr;  // NOTE: not implemented

  if (!transA_ && !transB_) {
    return (*GetOp<MLFloat16, Fp8T, MLFloat16, BlasOp::NonTrans, BlasOp::NonTrans>())(&params);
  } else if (transA_ && !transB_) {
    ORT_NOT_IMPLEMENTED("transA is not implemented");
  } else if (!transA_ && transB_) {
    return (*GetOp<MLFloat16, Fp8T, MLFloat16, BlasOp::NonTrans, BlasOp::Trans>())(&params);
  } else if (transA_ && transB_) {
    ORT_NOT_IMPLEMENTED("transA & transB is not implemented");
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unreachable");
}
#define GEMM_FLOAT8_CONSTRAINTS BuildKernelDefConstraints<MLFloat16, Float8E4M3FN, Float8E4M3FNUZ>()
#else
#define GEMM_FLOAT8_CONSTRAINTS BuildKernelDefConstraints<MLFloat16>()
#endif

ONNX_OPERATOR_KERNEL_EX(
    GemmFloat8,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("TA", GEMM_FLOAT8_CONSTRAINTS)
        .TypeConstraint("TB", GEMM_FLOAT8_CONSTRAINTS)
        .TypeConstraint("TR", BuildKernelDefConstraints<MLFloat16>())
        .TypeConstraint("TS", BuildKernelDefConstraints<float>()),
    GemmFloat8);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
