// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(_M_AMD64)
#include <intrin.h>
#endif

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if defined(_M_AMD64) || defined(__x86_64__)
#include <cstdint>
#endif

namespace onnxruntime {

namespace concurrency {

// Intrinsic to use in spin-loops

inline void SpinPause() {
#if defined(_M_AMD64) || defined(__x86_64__)
  _mm_pause();
#endif
}

inline void SpinTPAUSE() {
#if defined(_M_AMD64) || defined(__x86_64__)
   const std::uint64_t spin_delay_cycles = 2000;
  _tpause(0x0, __rdtsc() + spin_delay_cycles);
#endif
}

}  // namespace concurrency

}  // namespace onnxruntime
