// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(_M_AMD64)
#include <intrin.h>
#endif

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if defined(_WIN32) && defined(_MSC_VER) defined(_M_ARM)
extern "C" void YieldProcessor();
#endif

namespace onnxruntime {

namespace concurrency {

// Intrinsic to use in spin-loops
inline void SpinPause() {
#if defined(_M_AMD64) || defined(__x86_64__)
  _mm_pause();

#elif defined(_M_ARM64) || defined(_M_ARM) || defined(__arm__) || defined(__aarch64__)

#if defined(_WIN32) && defined(_MSC_VER) && defined(_M_ARM)
  YieldProcessor();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#else
  asm volatile("nop" ::: "memory")
#endif

#elif defined(__powerpc__) || defined(__POWERPC__)
  asm volatile("" ::: "memory");
#endif
}

}  // namespace concurrency

}  // namespace onnxruntime
