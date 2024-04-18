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
#include "core/common/cpuid_info.h"
#endif

namespace onnxruntime {

#if defined(_M_AMD64) || defined(__x86_64__)
const bool tpause = CPUIDInfo::GetCPUIDInfo().HasTPAUSE();
const std::uint64_t spin_delay_cycles = 2000;
#endif

namespace concurrency {

// Intrinsic to use in spin-loops
inline void SpinPause() {
#if defined(_M_AMD64) || defined(__x86_64__)
if(tpause) {
  _tpause(0x0, __rdtsc() + spin_delay_cycles);
}
else{
  _mm_pause();
}
#endif
}

}  // namespace concurrency

}  // namespace onnxruntime

