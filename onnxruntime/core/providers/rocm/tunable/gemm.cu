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

template <typename T, typename ScalarT>
inline GEMM(T, ScalarT) {
  GemmParams<T> params;
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
  if constexpr (!std::is_same_v<T, ScalarT> && std::is_same_v<ScalarT, float>) {
    params.beta = ToHipType<T>::FromFloat(std::forward<T>(beta));
  } else {
    params.beta = beta;
  }
  params.c = c;
  params.ldc = ldc;

  // TODO: current implementation for beta != 0 will cause repeatedly inplace update in C buffer. Skip them for now.
  if (tunable) {
    if (opa == BlasOp::N && opb == BlasOp::N) {
      static internal::GemmTunableOp<T, internal::Row, internal::Row> gemm{};
      gemm.EnableTuning();
      return gemm(&params);
    } else if (opa == BlasOp::T && opb == BlasOp::N) {
      static internal::GemmTunableOp<T, internal::Col, internal::Row> gemm{};
      gemm.EnableTuning();
      return gemm(&params);
    } else if (opa == BlasOp::N && opb == BlasOp::T) {
      static internal::GemmTunableOp<T, internal::Row, internal::Col> gemm{};
      gemm.EnableTuning();
      return gemm(&params);
    } else /*if (opa == BlasOp::T && opb == BlasOp::T)*/ {
      static internal::GemmTunableOp<T, internal::Col, internal::Col> gemm{};
      gemm.EnableTuning();
      return gemm(&params);
    }
  }

  return internal::RocBlasGemmOp(&params);
}

#define CALL_GEMM(T, ScalarT)               \
  Gemm<T, ScalarT>(tunable, stream, handle, \
                   opa, opb,                \
                   m, n, k,                 \
                   alpha, a, lda, b, ldb,   \
                   beta, c, ldc)

// clang-format off
GEMM(double,   double  ) { return CALL_GEMM(double,   double  ); }
GEMM(float,    float   ) { return CALL_GEMM(float,    float   ); }
GEMM(half,     half    ) { return CALL_GEMM(half,     half    ); }
GEMM(BFloat16, BFloat16) { return CALL_GEMM(BFloat16, BFloat16); }
GEMM(double,   float   ) { return CALL_GEMM(double,   float   ); }
GEMM(half,     float   ) { return CALL_GEMM(half,     float   ); }
GEMM(BFloat16, float   ) { return CALL_GEMM(BFloat16, float   ); }
// clang-format on

#undef CALL_GEMM

}  // namespace row_major

namespace column_major {

#define CALL_GEMM_WITH_AB_SWAPPED(T, ScalarT)          \
  row_major::Gemm<T, ScalarT>(tunable, stream, handle, \
                              opb, opa,                \
                              n, m, k,                 \
                              alpha, b, ldb, a, lda,   \
                              beta, c, ldc)

// clang-format off
GEMM(double,   double  ) { return CALL_GEMM_WITH_AB_SWAPPED(double,   double  ); }
GEMM(float,    float   ) { return CALL_GEMM_WITH_AB_SWAPPED(float,    float   ); }
GEMM(half,     half    ) { return CALL_GEMM_WITH_AB_SWAPPED(half,     half    ); }
GEMM(BFloat16, BFloat16) { return CALL_GEMM_WITH_AB_SWAPPED(BFloat16, BFloat16); }
GEMM(double,   float   ) { return CALL_GEMM_WITH_AB_SWAPPED(double,   float   ); }
GEMM(half,     float   ) { return CALL_GEMM_WITH_AB_SWAPPED(half,     float   ); }
GEMM(BFloat16, float   ) { return CALL_GEMM_WITH_AB_SWAPPED(BFloat16, float   ); }
// clang-format on

#undef CALL_GEMM_WITH_AB_SWAPPED

}  // namespace column_major

}  // namespace blas
}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
