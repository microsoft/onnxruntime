#pragma once

// default to gemmlowp when building for arm devices
#ifndef USE_GEMMLOWP
#if defined(_M_ARM64) || defined(__aarch64__)
#define USE_GEMMLOWP
#endif
#if defined(_M_ARM) || defined(__arm__)
#define USE_GEMMLOWP
#endif
#endif

#ifdef USE_GEMMLOWP
#include "core/util/gemmlowp_common.h"
#else
#include "core/mlas/inc/mlas.h"
#endif
#include "core/platform/threadpool.h"
#include <mutex>
#include <thread>

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
    concurrency::ThreadPool* thread_pool);

}  // namespace onnxruntime
