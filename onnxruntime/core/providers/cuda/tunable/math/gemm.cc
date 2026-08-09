// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tunable/math/gemm.h"

#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace cuda {
namespace tunable {

template <typename T>
GemmParams<T>::GemmParams(int m, int n, int k, bool trans_a, bool trans_b, float alpha, float beta,
                          const Gemm<T>* gemm_kernel, OpKernelContext* ctx)
    : OpParams(gemm_kernel->GetTuningContext(), ctx->GetComputeStream()),
      trans_a_(trans_a),
      trans_b_(trans_b),
      alpha_(alpha),
      beta_(beta),
      m_(m),
      n_(n),
      k_(k),
      gemm_kernel_(gemm_kernel),
      ctx_(ctx) {
  const auto* b = ctx->Input<Tensor>(2);
  bm_ = gsl::narrow_cast<int>(beta_ == 0.0f ? 0 : (b->Shape().NumDimensions() > 1 ? b->Shape()[0] : 1));
  bn_ = gsl::narrow_cast<int>(
      beta_ == 0.0f
          ? 0
          : (b->Shape().NumDimensions() > 1 ? b->Shape()[1] : (b->Shape().NumDimensions() > 0 ? b->Shape()[0] : 1)));

#ifdef ENABLE_TRITON
  const auto* x = ctx->Input<Tensor>(0);
  has_triton_support_ = contrib::IsTritonOpExecutorInitialized() &&
                        (std::is_same<T, MLFloat16>::value || std::is_same<T, float>::value) &&
                        x->Shape().NumDimensions() > 1;
#endif
}

namespace {

template <typename T>
common::Status DefaultGemmOp(const GemmParams<T>* params) {
  return params->gemm_kernel_->ComputeDefault(params->ctx_, params->m_, params->n_, params->k_);
}

#ifdef ENABLE_TRITON
template <typename T>
common::Status TritonGemmOp(const GemmParams<T>* params) {
  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!params->has_triton_support_);
  size_t input_count = params->beta_ == 0.0f ? 2 : 3;
  size_t output_count = 1;
  std::string func_name = params->beta_ == 0.0f ? "triton_matmul_out" : "triton_gemm_out";
  InlinedHashMap<std::string, std::pair<std::string, int>> kwargs;
  if (params->trans_a_) kwargs.insert({"trans_a", {"true", ONNX_NAMESPACE::TensorProto_DataType_BOOL}});
  if (params->trans_b_) kwargs.insert({"trans_b", {"true", ONNX_NAMESPACE::TensorProto_DataType_BOOL}});
  if (params->alpha_ != 1.0f) {
    kwargs.insert({"alpha", {std::to_string(params->alpha_), ONNX_NAMESPACE::TensorProto_DataType_FLOAT}});
  }
  if (params->beta_ != 0.0f && params->beta_ != 1.0f) {
    kwargs.insert({"beta", {std::to_string(params->beta_), ONNX_NAMESPACE::TensorProto_DataType_FLOAT}});
  }
  return contrib::ExecuteTritonOpByFuncName(params->ctx_, func_name, input_count, output_count, kwargs);
}
#endif

template <typename T>
class GemmTunableOp : public TunableOp<GemmParams<T>> {
 public:
  GemmTunableOp() {
    this->RegisterOp(DefaultGemmOp<T>);
#ifdef ENABLE_TRITON
    this->RegisterOp(TritonGemmOp<T>);
#endif
  }
};

}  // namespace

template <typename T>
inline common::Status TunableGemm(int m, int n, int k, bool trans_a, bool trans_b, float alpha, float beta,
                                  const Gemm<T>* gemm_kernel, OpKernelContext* ctx) {
  GemmParams<T> params(m, n, k, trans_a, trans_b, alpha, beta, gemm_kernel, ctx);
  if (params.tuning_ctx->IsTunableOpEnabled()) {
    static GemmTunableOp<T> gemm{};
    return gemm(&params);
  }

  return DefaultGemmOp(&params);
}

#define SPECIALIZE_TUNABLE_GEMM(T)                                                                                 \
  template common::Status TunableGemm<T>(int m, int n, int k, bool trans_a, bool trans_b, float alpha, float beta, \
                                         const Gemm<T>* gemm_kernel, OpKernelContext* ctx);

SPECIALIZE_TUNABLE_GEMM(float)
SPECIALIZE_TUNABLE_GEMM(double)
SPECIALIZE_TUNABLE_GEMM(MLFloat16)
SPECIALIZE_TUNABLE_GEMM(BFloat16)

#undef SPECIALIZE_TUNABLE_GEMM

}  // namespace tunable
}  // namespace cuda
}  // namespace onnxruntime
