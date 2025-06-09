// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/spin_pause.h"

#if defined(_M_AMD64)
#include <intrin.h>
#endif

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if defined(_M_AMD64) || defined(__x86_64__)
#include "core/common/cpuid_info.h"
#if defined(__linux__)
#include <x86intrin.h>
#include <immintrin.h>
#endif
#endif

namespace onnxruntime {
namespace concurrency {

// Intrinsic to use in spin-loops
void SpinPause() {
#if (defined(_M_AMD64) || defined(__x86_64__)) && \
    !defined(_M_ARM64EC) &&                       \
    !defined(__ANDROID__) &&                      \
    !defined(__APPLE__)

  static const bool has_tpause = CPUIDInfo::GetCPUIDInfo().HasTPAUSE();
  static constexpr uint64_t tpause_spin_delay_cycles = 1000;
  if (has_tpause) {
#if defined(_WIN32)
    _tpause(0x0, __rdtsc() + tpause_spin_delay_cycles);
#elif defined(__linux__)
    __builtin_ia32_tpause(0x0, __rdtsc() + tpause_spin_delay_cycles);
#else
    _mm_pause();
#endif
  } else {
    _mm_pause();
  }
#endif
}

}  // namespace concurrency
}  // namespace onnxruntime
