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
#if defined(__linux__)
#include <x86intrin.h>
#include <immintrin.h>
#include <cstdlib>
#elif defined(_WIN32)
#include <Windows.h>
#endif
#include <cstdint>
#include <string>
#include "core/common/cpuid_info.h"
#endif

namespace onnxruntime {

#if defined(_M_AMD64) || defined(__x86_64__)
static const bool tpause = CPUIDInfo::GetCPUIDInfo().HasTPAUSE();
static const std::uint64_t spin_delay_cycles = 1000;
#endif

namespace concurrency {

// Intrinsic to use in spin-loops
void SpinPause() {
#if defined(_M_AMD64) || defined(__x86_64__)
#if defined(_WIN32)
  if (tpause) {
    _tpause(0x0, __rdtsc() + spin_delay_cycles);
  } else {
    _mm_pause();
  }
#elif defined(__linux__)
  if (__builtin_cpu_supports("waitpkg")) {
    __builtin_ia32_tpause(0x0, __rdtsc() + spin_delay_cycles);
  } else {
    _mm_pause();
  }
#else
  _mm_pause();
#endif
#endif
}

}  // namespace concurrency
}  // namespace onnxruntime
