// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define _GEMM_FASTGELU_H_KEEP_SIGNATURE_DEFINES
#include "contrib_ops/rocm/bert/gemm_fast_gelu_impl.h"

#include <type_traits>
#include <utility>

#include "contrib_ops/rocm/bert/gemm_fast_gelu_tunable.cuh"
#include "core/providers/rocm/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {
namespace blas {

namespace row_major {

template <typename T, typename ScalarT>
inline GEMMFASTGELU(T, ScalarT) {
  GemmFastGeluParams<T> params;
  params.tuning_ctx = tuning_ctx;
  params.stream = stream;
  params.handle = handle;

  params.opa = opa;
  params.opb = opb;
  params.m = m;
  params.n = n;
  params.k = k;
  if constexpr (!std::is_same_v<T, ScalarT> && std::is_same_v<ScalarT, float>) {
    params.alpha = ToHipType<T>::FromFloat(std::forward<T>(alpha));
  } else {
    params.alpha = alpha;
  }
  params.a = a;
  params.lda = lda;
  params.b = b;
  params.ldb = ldb;
  params.bias = bias;
  if constexpr (!std::is_same_v<T, ScalarT> && std::is_same_v<ScalarT, float>) {
    params.beta = ToHipType<T>::FromFloat(std::forward<T>(beta));
  } else {
    params.beta = beta;
  }
  params.c = c;
  params.ldc = ldc;

  if (tuning_ctx->IsTunableOpEnabled()) {
    if (opa == BlasOp::N && opb == BlasOp::N) {
      static internal::GemmFastGeluTunableOp<T, BlasOp::N, BlasOp::N> gemm_fast_gelu{};
      return gemm_fast_gelu(&params);
    } else if (opa == BlasOp::T && opb == BlasOp::N) {
      static internal::GemmFastGeluTunableOp<T, BlasOp::T, BlasOp::N> gemm_fast_gelu{};
      return gemm_fast_gelu(&params);
    } else if (opa == BlasOp::N && opb == BlasOp::T) {
      static internal::GemmFastGeluTunableOp<T, BlasOp::N, BlasOp::T> gemm_fast_gelu{};
      return gemm_fast_gelu(&params);
    } else /*if (opa == BlasOp::T && opb == BlasOp::T)*/ {
      static internal::GemmFastGeluTunableOp<T, BlasOp::T, BlasOp::T> gemm_fast_gelu{};
      return gemm_fast_gelu(&params);
    }
  }

  return internal::GemmFastGeluUnfused(&params);
}

#define CALL_GEMMFASTGELU(T, ScalarT)                   \
  GemmFastGelu<T, ScalarT>(tuning_ctx, stream, handle,  \
                           opa, opb,                    \
                           m, n, k,                     \
                           alpha, a, lda, b, ldb, bias, \
                           beta, c, ldc)

// clang-format off
GEMMFASTGELU(float,    float   ) { return CALL_GEMMFASTGELU(float,    float   ); }
GEMMFASTGELU(half,     half    ) { return CALL_GEMMFASTGELU(half,     half    ); }
GEMMFASTGELU(BFloat16, BFloat16) { return CALL_GEMMFASTGELU(BFloat16, BFloat16); }
GEMMFASTGELU(half,     float   ) { return CALL_GEMMFASTGELU(half,     float   ); }
GEMMFASTGELU(BFloat16, float   ) { return CALL_GEMMFASTGELU(BFloat16, float   ); }
// clang-format on

#undef CALL_GEMMFASTGELU

}  // namespace row_major

}  // namespace blas
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
