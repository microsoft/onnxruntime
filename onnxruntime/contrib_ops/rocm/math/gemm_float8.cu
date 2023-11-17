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

    tunable_op_fp8e4m3fn_fp16_fp16_ = std::make_unique<decltype(tunable_op_fp8e4m3fn_fp16_fp16_)::element_type>();
    tunable_op_fp8e4m3fnuz_fp16_fp16_ = std::make_unique<decltype(tunable_op_fp8e4m3fnuz_fp16_fp16_)::element_type>();
    tunable_op_fp16_fp8e4m3fn_fp16_ = std::make_unique<decltype(tunable_op_fp16_fp8e4m3fn_fp16_)::element_type>();
    tunable_op_fp16_fp8e4m3fnuz_fp16_ = std::make_unique<decltype(tunable_op_fp16_fp8e4m3fnuz_fp16_)::element_type>();
  }
  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  template <typename Fp8T>
  Status ComputeFp8Fp16Fp16(OpKernelContext* ctx, const Tensor* A, const Tensor* scaleA, const Tensor* B, Tensor* C) const;
  template <typename Fp8T>
  Status ComputeFp16Fp8Fp16(OpKernelContext* ctx, const Tensor* A, const Tensor* B, const Tensor* scaleB, Tensor* C) const;

  template <typename Fp8T, bool IsAFp8>
  [[nodiscard]] inline auto& GetOp() const {
    if constexpr (std::is_same_v<Fp8T, Float8E4M3FN>) {
      if constexpr (IsAFp8) {
        return tunable_op_fp8e4m3fn_fp16_fp16_;
      } else {
        return tunable_op_fp16_fp8e4m3fn_fp16_;
      }
    } else if constexpr (std::is_same_v<Fp8T, Float8E4M3FNUZ>) {
      if constexpr (IsAFp8) {
        return tunable_op_fp8e4m3fnuz_fp16_fp16_;
      } else {
        return tunable_op_fp16_fp8e4m3fnuz_fp16_;
      }
    }
  }

  float alpha_;
  float beta_;
  bool transA_;
  bool transB_;
  int64_t dtype_;

  std::unique_ptr<F8GemmTunableOp<Float8E4M3FN, MLFloat16, MLFloat16, internal::Row, internal::Row>> tunable_op_fp8e4m3fn_fp16_fp16_;
  std::unique_ptr<F8GemmTunableOp<Float8E4M3FNUZ, MLFloat16, MLFloat16, internal::Row, internal::Row>> tunable_op_fp8e4m3fnuz_fp16_fp16_;
  std::unique_ptr<F8GemmTunableOp<MLFloat16, Float8E4M3FN, MLFloat16, internal::Row, internal::Row>> tunable_op_fp16_fp8e4m3fn_fp16_;
  std::unique_ptr<F8GemmTunableOp<MLFloat16, Float8E4M3FNUZ, MLFloat16, internal::Row, internal::Row>> tunable_op_fp16_fp8e4m3fnuz_fp16_;
};

Status GemmFloat8::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* A = ctx->Input<Tensor>(0);
  const Tensor* B = ctx->Input<Tensor>(1);
  const Tensor* C = ctx->Input<Tensor>(2);  // bias
  const Tensor* scale_a = ctx->Input<Tensor>(3);
  const Tensor* scale_b = ctx->Input<Tensor>(4);
  const Tensor* scale_y = ctx->Input<Tensor>(5);

  auto a_shape = A->Shape();
  auto b_shape = B->Shape();
  ORT_ENFORCE(a_shape.NumDimensions() >= 2 && b_shape.NumDimensions() == 2);  // is in form of input @ weight
  ORT_ENFORCE(a_shape[a_shape.NumDimensions() - 1] == b_shape[0]);            // k is compatiable

  TensorShapeVector output_shape = a_shape.AsShapeVector();
  output_shape[output_shape.size() - 1] = b_shape[b_shape.NumDimensions() - 1];
  Tensor* Y = ctx->Output(0, output_shape);

  ORT_ENFORCE(!transA_ && !transB_, "ROCm GemmFloat8 does not support input transpose");
  ORT_ENFORCE(dtype_ == onnx::TensorProto_DataType_FLOAT16, "ROCm GemmFloat8 only supports output float16");
  ORT_ENFORCE(C == nullptr, "ROCm GemmFloat8 does not support bias input");
  ORT_ENFORCE(scale_y == nullptr, "ROCm GemmFloat8 does not support output scaling");

  if (A->IsDataType<Float8E4M3FN>()) {
    return ComputeFp8Fp16Fp16<Float8E4M3FN>(ctx, A, scale_a, B, Y);
  } else if (A->IsDataType<Float8E4M3FNUZ>()) {
    return ComputeFp8Fp16Fp16<Float8E4M3FNUZ>(ctx, A, scale_a, B, Y);
  } else if (B->IsDataType<Float8E4M3FN>()) {
    return ComputeFp16Fp8Fp16<Float8E4M3FN>(ctx, A, B, scale_b, Y);
  } else if (B->IsDataType<Float8E4M3FNUZ>()) {
    return ComputeFp16Fp8Fp16<Float8E4M3FNUZ>(ctx, A, B, scale_b, Y);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unhandled type combination of GemmFloat8");
}

template <typename Fp8T>
Status GemmFloat8::ComputeFp8Fp16Fp16(OpKernelContext* ctx, const Tensor* A, const Tensor* scale_a, const Tensor* B, Tensor* C) const {
  ORT_ENFORCE(A->IsDataType<Fp8T>() && scale_a->IsDataType<float>() && B->IsDataType<MLFloat16>());

  auto a_shape = A->Shape();
  auto b_shape = B->Shape();

  auto m = a_shape.Slice(0, a_shape.NumDimensions() - 1).Size();
  auto k = a_shape[a_shape.NumDimensions() - 1];
  auto n = b_shape[b_shape.NumDimensions() - 1];

  onnxruntime::rocm::tunable::blas::GemmFloat8Params<Fp8T, MLFloat16, MLFloat16> params{};
  params.tuning_ctx = GetTuningContext();
  params.stream = ctx->GetComputeStream();
  params.handle = GetRocblasHandle(ctx);
  params.opa = tunable::blas::BlasOp::NonTrans;
  params.opb = tunable::blas::BlasOp::NonTrans;

  params.m = m;
  params.n = n;
  params.k = k;

  params.a = static_cast<const Fp8T*>(A->DataRaw());
  params.lda = k;
  params.scale_a = alpha_;
  params.scale_a_dev = static_cast<const float*>(scale_a->DataRaw());

  params.b = static_cast<const MLFloat16*>(B->DataRaw());
  params.ldb = n;
  params.scale_b = 1.0f;         // NOTE: not used
  params.scale_b_dev = nullptr;  // NOTE: not used

  params.c = static_cast<MLFloat16*>(C->MutableDataRaw());
  params.ldc = n;
  params.scale_c = 1.0f;         // NOTE: not implemented
  params.scale_c_dev = nullptr;  // NOTE: not implemented

  return (*GetOp<Fp8T, true>())(&params);
}

template <typename Fp8T>
Status GemmFloat8::ComputeFp16Fp8Fp16(OpKernelContext* ctx, const Tensor* A, const Tensor* B, const Tensor* scale_b, Tensor* C) const {
  ORT_ENFORCE(A->IsDataType<MLFloat16>() && B->IsDataType<Fp8T>() && scale_b->IsDataType<float>());

  auto a_shape = A->Shape();
  auto b_shape = B->Shape();

  auto m = a_shape.Slice(0, a_shape.NumDimensions() - 1).Size();
  auto k = a_shape[a_shape.NumDimensions() - 1];
  auto n = b_shape[b_shape.NumDimensions() - 1];

  onnxruntime::rocm::tunable::blas::GemmFloat8Params<MLFloat16, Fp8T, MLFloat16> params{};
  params.tuning_ctx = GetTuningContext();
  params.stream = ctx->GetComputeStream();
  params.handle = GetRocblasHandle(ctx);
  params.opa = tunable::blas::BlasOp::NonTrans;
  params.opb = tunable::blas::BlasOp::NonTrans;

  params.m = m;
  params.n = n;
  params.k = k;

  params.a = static_cast<const MLFloat16*>(A->DataRaw());
  params.lda = k;
  params.scale_a = 1.0f;         // NOTE: not used
  params.scale_a_dev = nullptr;  // NOTE: not used

  params.b = static_cast<const Fp8T*>(B->DataRaw());
  params.ldb = n;
  params.scale_b = alpha_;
  params.scale_b_dev = static_cast<const float*>(scale_b->DataRaw());

  params.c = static_cast<MLFloat16*>(C->MutableDataRaw());
  params.ldc = n;
  params.scale_c = 1.0f;         // NOTE: not implemented
  params.scale_c_dev = nullptr;  // NOTE: not implemented

  return (*GetOp<Fp8T, false>())(&params);
}

ONNX_OPERATOR_KERNEL_EX(
    GemmFloat8,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("TA", BuildKernelDefConstraints<MLFloat16, Float8E4M3FN, Float8E4M3FNUZ>())
        .TypeConstraint("TB", BuildKernelDefConstraints<MLFloat16, Float8E4M3FN, Float8E4M3FNUZ>())
        .TypeConstraint("TR", BuildKernelDefConstraints<MLFloat16>())
        .TypeConstraint("TS", BuildKernelDefConstraints<float>()),
    GemmFloat8);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
