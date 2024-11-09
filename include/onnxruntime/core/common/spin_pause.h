// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(_M_AMD64)
#include <intrin.h>
#endif

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM))
extern "C" void YieldProcessor();
#endif

namespace onnxruntime {

namespace concurrency {

// Intrinsic to use in spin-loops
inline void SpinPause() {
#if defined(_M_AMD64) || defined(__x86_64__)
  _mm_pause();

#elif defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM))
  YieldProcessor();

// yield is supported from ARMv6k onwards.
#elif defined(__aarch64__) || (defined(__ARM_ARCH) && __ARM_ARCH >= 7)
  asm volatile("yield" ::: "memory");
#endif
}

}  // namespace concurrency

}  // namespace onnxruntime
