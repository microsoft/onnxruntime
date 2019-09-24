// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/util/qmath.h"
#include "core/common/common.h"

namespace onnxruntime {

void QGemmu8u8_s32(
    int M,
    int N,
    int K,
    const uint8_t* lhs_data,
    int lda,
    const uint8_t lhs_offset,
    const uint8_t* rhs_data,
    int ldb,
    const uint8_t rhs_offset,
    int32_t* result_data,
    int ldc,
    concurrency::ThreadPool* thread_pool) {
#ifdef USE_GEMMLOWP

  ORT_ENFORCE(lda == K && ldb == N &&  ldc == N, "For gemmlowp only RowMajor*RowMajor=RowMajor format is supported");

  GemmlowpMultiplyu8u8_s32(lhs_data, rhs_data, result_data, lhs_offset, rhs_offset, M, N, K, thread_pool);  

#else
  MlasQgemm(M, N, K, lhs_data, lda, lhs_offset, rhs_data, ldb, rhs_offset, result_data, ldc, thread_pool);

#endif
}
}  // namespace onnxruntime