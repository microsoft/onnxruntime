// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/threadpool.h"

#if defined(_M_AMD64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#define MLAS_SUPPORTS_GEMM_U8X8
#endif

namespace onnxruntime {

template <typename LeftScalar, typename RightScalar, typename OutputScalar>
void QGemm(
    int M,
    int N,
    int K,
    const LeftScalar* lhs_data,
    int lda,
    const LeftScalar lhs_offset,
    const RightScalar* rhs_data,
    int ldb,
    const RightScalar rhs_offset,
    OutputScalar* result_data,
    int ldc,
    concurrency::ThreadPool* thread_pool);
}  // namespace onnxruntime
