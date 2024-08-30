// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define _GEMM_H_KEEP_SIGNATURE_DEFINES
#include "core/providers/rocm/tunable/gemm.h"

#include <type_traits>
#include <utility>

#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/rocm/tunable/gemm_rocblas.h"
#include "core/providers/rocm/tunable/gemm_tunable.cuh"

namespace onnxruntime {
namespace rocm {
namespace tunable {
namespace blas {

namespace row_major {

namespace {
// a simple utility function that normalize alpha or beta to the desired datatype by an optional casting
template <typename DesiredT, typename ScalarT>
inline DesiredT NormalizeScalar(ScalarT v) {
  if constexpr (!std::is_same_v<DesiredT, ScalarT> && std::is_same_v<ScalarT, float>) {
    return ToHipType<DesiredT>::FromFloat(std::forward<DesiredT>(v));
  } else {
    return v;
  }
}
}  // namespace

template <typename T, typename ScalarT>
inline GEMM(T, ScalarT) {
  GemmParams<T> params;
  params.tuning_ctx = tuning_ctx;
  params.stream = stream;
  params.handle = handle;

  params.opa = opa;
  params.opb = opb;
  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = NormalizeScalar<T>(alpha);
  params.a = a;
  params.lda = lda;
  params.b = b;
  params.ldb = ldb;
  params.beta = NormalizeScalar<T>(beta);
  params.c = c;
  params.ldc = ldc;

  if (tuning_ctx->IsTunableOpEnabled()) {
    if (opa == BlasOp::N && opb == BlasOp::N) {
      static internal::GemmTunableOp<T, BlasOp::N, BlasOp::N> gemm{};
      return gemm(&params);
    } else if (opa == BlasOp::T && opb == BlasOp::N) {
      static internal::GemmTunableOp<T, BlasOp::T, BlasOp::N> gemm{};
      return gemm(&params);
    } else if (opa == BlasOp::N && opb == BlasOp::T) {
      static internal::GemmTunableOp<T, BlasOp::N, BlasOp::T> gemm{};
      return gemm(&params);
    } else /*if (opa == BlasOp::T && opb == BlasOp::T)*/ {
      static internal::GemmTunableOp<T, BlasOp::T, BlasOp::T> gemm{};
      return gemm(&params);
    }
  }

  return internal::RocBlasGemmOp(&params);
}

template <typename T, typename ScalarT>
inline BATCHED_GEMM(T, ScalarT) {
  BatchedGemmParams<T> params;
  params.tuning_ctx = tuning_ctx;
  params.stream = stream;
  params.handle = handle;

  params.opa = opa;
  params.opb = opb;
  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = NormalizeScalar<T>(alpha);
  params.as = as;
  params.lda = lda;
  params.bs = bs;
  params.ldb = ldb;
  params.beta = NormalizeScalar<T>(beta);
  params.cs = cs;
  params.ldc = ldc;
  params.batch = batch;

  if (tuning_ctx->IsTunableOpEnabled()) {
    if (opa == BlasOp::N && opb == BlasOp::N) {
      static internal::BatchedGemmTunableOp<T, BlasOp::N, BlasOp::N> gemm{};
      return gemm(&params);
    } else if (opa == BlasOp::T && opb == BlasOp::N) {
      static internal::BatchedGemmTunableOp<T, BlasOp::T, BlasOp::N> gemm{};
      return gemm(&params);
    } else if (opa == BlasOp::N && opb == BlasOp::T) {
      static internal::BatchedGemmTunableOp<T, BlasOp::N, BlasOp::T> gemm{};
      return gemm(&params);
    } else /*if (opa == BlasOp::T && opb == BlasOp::T)*/ {
      static internal::BatchedGemmTunableOp<T, BlasOp::T, BlasOp::T> gemm{};
      return gemm(&params);
    }
  }

  return internal::RocBlasBatchedGemmOp(&params);
}

template <typename T, typename ScalarT>
inline STRIDED_BATCHED_GEMM(T, ScalarT) {
  StridedBatchedGemmParams<T> params;
  params.tuning_ctx = tuning_ctx;
  params.stream = stream;
  params.handle = handle;

  params.opa = opa;
  params.opb = opb;
  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = NormalizeScalar<T>(alpha);
  params.a = a;
  params.lda = lda;
  params.stride_a = stride_a;
  params.b = b;
  params.ldb = ldb;
  params.stride_b = stride_b;
  params.beta = NormalizeScalar<T>(beta);
  params.c = c;
  params.ldc = ldc;
  params.stride_c = stride_c;
  params.batch = batch;

  if (tuning_ctx->IsTunableOpEnabled()) {
    if (opa == BlasOp::N && opb == BlasOp::N) {
      static internal::StridedBatchedGemmTunableOp<T, BlasOp::N, BlasOp::N> gemm{};
      return gemm(&params);
    } else if (opa == BlasOp::T && opb == BlasOp::N) {
      static internal::StridedBatchedGemmTunableOp<T, BlasOp::T, BlasOp::N> gemm{};
      return gemm(&params);
    } else if (opa == BlasOp::N && opb == BlasOp::T) {
      static internal::StridedBatchedGemmTunableOp<T, BlasOp::N, BlasOp::T> gemm{};
      return gemm(&params);
    } else /*if (opa == BlasOp::T && opb == BlasOp::T)*/ {
      static internal::StridedBatchedGemmTunableOp<T, BlasOp::T, BlasOp::T> gemm{};
      return gemm(&params);
    }
  }

  return internal::RocBlasStridedBatchedGemmOp(&params);
}

#define CALL_GEMM(T, ScalarT)                  \
  Gemm<T, ScalarT>(tuning_ctx, stream, handle, \
                   opa, opb,                   \
                   m, n, k,                    \
                   alpha, a, lda, b, ldb,      \
                   beta, c, ldc)

#define CALL_BATCHED_GEMM(T, ScalarT) \
  BatchedGemm<T, ScalarT>(            \
      tuning_ctx, stream, handle,     \
      opa, opb,                       \
      m, n, k,                        \
      alpha, as, lda, bs, ldb,        \
      beta, cs, ldc, batch)

#define CALL_STRIDED_BATCHED_GEMM(T, ScalarT) \
  StridedBatchedGemm<T, ScalarT>(             \
      tuning_ctx, stream, handle,             \
      opa, opb,                               \
      m, n, k,                                \
      alpha,                                  \
      a, lda, stride_a,                       \
      b, ldb, stride_b,                       \
      beta, c, ldc, stride_c,                 \
      batch)

// clang-format off
GEMM(double,   double  ) { return CALL_GEMM(double,   double  ); }
GEMM(float,    float   ) { return CALL_GEMM(float,    float   ); }
GEMM(half,     half    ) { return CALL_GEMM(half,     half    ); }
GEMM(BFloat16, BFloat16) { return CALL_GEMM(BFloat16, BFloat16); }
GEMM(double,   float   ) { return CALL_GEMM(double,   float   ); }
GEMM(half,     float   ) { return CALL_GEMM(half,     float   ); }
GEMM(BFloat16, float   ) { return CALL_GEMM(BFloat16, float   ); }

BATCHED_GEMM(double,   double  ) { return CALL_BATCHED_GEMM(double,   double  ); }
BATCHED_GEMM(float,    float   ) { return CALL_BATCHED_GEMM(float,    float   ); }
BATCHED_GEMM(half,     half    ) { return CALL_BATCHED_GEMM(half,     half    ); }
BATCHED_GEMM(BFloat16, BFloat16) { return CALL_BATCHED_GEMM(BFloat16, BFloat16); }
BATCHED_GEMM(double,   float   ) { return CALL_BATCHED_GEMM(double,   float   ); }
BATCHED_GEMM(half,     float   ) { return CALL_BATCHED_GEMM(half,     float   ); }
BATCHED_GEMM(BFloat16, float   ) { return CALL_BATCHED_GEMM(BFloat16, float   ); }

STRIDED_BATCHED_GEMM(double,   double  ) { return CALL_STRIDED_BATCHED_GEMM(double,   double  ); }
STRIDED_BATCHED_GEMM(float,    float   ) { return CALL_STRIDED_BATCHED_GEMM(float,    float   ); }
STRIDED_BATCHED_GEMM(half,     half    ) { return CALL_STRIDED_BATCHED_GEMM(half,     half    ); }
STRIDED_BATCHED_GEMM(BFloat16, BFloat16) { return CALL_STRIDED_BATCHED_GEMM(BFloat16, BFloat16); }
STRIDED_BATCHED_GEMM(double,   float   ) { return CALL_STRIDED_BATCHED_GEMM(double,   float   ); }
STRIDED_BATCHED_GEMM(half,     float   ) { return CALL_STRIDED_BATCHED_GEMM(half,     float   ); }
STRIDED_BATCHED_GEMM(BFloat16, float   ) { return CALL_STRIDED_BATCHED_GEMM(BFloat16, float   ); }
// clang-format on

#undef CALL_GEMM
#undef CALL_BATCHED_GEMM
#undef CALL_STRIDED_BATCHED_GEMM

}  // namespace row_major

namespace column_major {

#define CALL_GEMM_WITH_AB_SWAPPED(T, ScalarT)             \
  row_major::Gemm<T, ScalarT>(tuning_ctx, stream, handle, \
                              opb, opa,                   \
                              n, m, k,                    \
                              alpha, b, ldb, a, lda,      \
                              beta, c, ldc)

#define CALL_BATCHED_GEMM_WITH_AB_SWAPPED(T, ScalarT) \
  row_major::BatchedGemm<T, ScalarT>(                 \
      tuning_ctx, stream, handle,                     \
      opb, opa,                                       \
      n, m, k,                                        \
      alpha, bs, ldb, as, lda,                        \
      beta, cs, ldc, batch)

#define CALL_STRIDED_BATCHED_GEMM_WITH_AB_SWAPPED(T, ScalarT) \
  row_major::StridedBatchedGemm<T, ScalarT>(                  \
      tuning_ctx, stream, handle,                             \
      opb, opa,                                               \
      n, m, k,                                                \
      alpha,                                                  \
      b, ldb, stride_b,                                       \
      a, lda, stride_a,                                       \
      beta,                                                   \
      c, ldc, stride_c,                                       \
      batch)

// clang-format off
GEMM(double,   double  ) { return CALL_GEMM_WITH_AB_SWAPPED(double,   double  ); }
GEMM(float,    float   ) { return CALL_GEMM_WITH_AB_SWAPPED(float,    float   ); }
GEMM(half,     half    ) { return CALL_GEMM_WITH_AB_SWAPPED(half,     half    ); }
GEMM(BFloat16, BFloat16) { return CALL_GEMM_WITH_AB_SWAPPED(BFloat16, BFloat16); }
GEMM(double,   float   ) { return CALL_GEMM_WITH_AB_SWAPPED(double,   float   ); }
GEMM(half,     float   ) { return CALL_GEMM_WITH_AB_SWAPPED(half,     float   ); }
GEMM(BFloat16, float   ) { return CALL_GEMM_WITH_AB_SWAPPED(BFloat16, float   ); }

BATCHED_GEMM(double,   double  ) { return CALL_BATCHED_GEMM_WITH_AB_SWAPPED(double,   double  ); }
BATCHED_GEMM(float,    float   ) { return CALL_BATCHED_GEMM_WITH_AB_SWAPPED(float,    float   ); }
BATCHED_GEMM(half,     half    ) { return CALL_BATCHED_GEMM_WITH_AB_SWAPPED(half,     half    ); }
BATCHED_GEMM(BFloat16, BFloat16) { return CALL_BATCHED_GEMM_WITH_AB_SWAPPED(BFloat16, BFloat16); }
BATCHED_GEMM(double,   float   ) { return CALL_BATCHED_GEMM_WITH_AB_SWAPPED(double,   float   ); }
BATCHED_GEMM(half,     float   ) { return CALL_BATCHED_GEMM_WITH_AB_SWAPPED(half,     float   ); }
BATCHED_GEMM(BFloat16, float   ) { return CALL_BATCHED_GEMM_WITH_AB_SWAPPED(BFloat16, float   ); }

STRIDED_BATCHED_GEMM(double,   double  ) { return CALL_STRIDED_BATCHED_GEMM_WITH_AB_SWAPPED(double,   double  ); }
STRIDED_BATCHED_GEMM(float,    float   ) { return CALL_STRIDED_BATCHED_GEMM_WITH_AB_SWAPPED(float,    float   ); }
STRIDED_BATCHED_GEMM(half,     half    ) { return CALL_STRIDED_BATCHED_GEMM_WITH_AB_SWAPPED(half,     half    ); }
STRIDED_BATCHED_GEMM(BFloat16, BFloat16) { return CALL_STRIDED_BATCHED_GEMM_WITH_AB_SWAPPED(BFloat16, BFloat16); }
STRIDED_BATCHED_GEMM(double,   float   ) { return CALL_STRIDED_BATCHED_GEMM_WITH_AB_SWAPPED(double,   float   ); }
STRIDED_BATCHED_GEMM(half,     float   ) { return CALL_STRIDED_BATCHED_GEMM_WITH_AB_SWAPPED(half,     float   ); }
STRIDED_BATCHED_GEMM(BFloat16, float   ) { return CALL_STRIDED_BATCHED_GEMM_WITH_AB_SWAPPED(BFloat16, float   ); }
// clang-format on

#undef CALL_GEMM_WITH_AB_SWAPPED
#undef CALL_BATCHED_GEMM_WITH_AB_SWAPPED
#undef CALL_STRIDED_BATCHED_GEMM_WITH_AB_SWAPPED

}  // namespace column_major

}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
