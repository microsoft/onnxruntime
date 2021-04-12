// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#define PLATFORM_X86
#endif

#if defined(PLATFORM_X86)
#include <memory>
#include <mutex>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__)
#include <cpuid.h>
#endif
#endif

#include "core/common/cpuid_info.h"

namespace onnxruntime {

#if defined(PLATFORM_X86)
static inline void GetCPUID(int function_id, int data[4]) {  // NOLINT
#if defined(_MSC_VER)
  __cpuid(reinterpret_cast<int*>(data), function_id);
#elif defined(__GNUC__)
  __cpuid(function_id, data[0], data[1], data[2], data[3]);
#endif
}

static inline int XGETBV() {
#if defined(_MSC_VER)
  return static_cast<int>(_xgetbv(0));
#elif defined(__GNUC__)
  int eax, edx;
  __asm__ volatile("xgetbv"
                   : "=a"(eax), "=d"(edx)
                   : "c"(0));
  return eax;
#endif
}
#endif  // PLATFORM_X86

CPUIDInfo::CPUIDInfo() noexcept {
#if defined(PLATFORM_X86)
  int data[4] = {-1};
  GetCPUID(0, data);

  int num_IDs = data[0];
  if (num_IDs >= 1) {
    GetCPUID(1, data);
    if (data[2] & (1 << 27)) {
      const int AVX_MASK = 0x6;
      const int AVX512_MASK = 0xE6;
      int value = XGETBV();
      bool has_sse2 = (data[3] & (1 << 26));
      has_sse3_ = (data[2] & 0x1);
      bool has_ssse3 = (data[2] & (1 << 9));
      has_avx_ = has_sse2 && has_ssse3 && (data[2] & (1 << 28)) && ((value & AVX_MASK) == AVX_MASK);
      bool has_avx512 = (value & AVX512_MASK) == AVX512_MASK;
      has_f16c_ = has_avx_ && (data[2] & (1 << 29)) && (data[3] & (1 << 26));

      if (num_IDs >= 7) {
        GetCPUID(7, data);
        has_avx2_ = has_avx_ && (data[1] & (1 << 5));
        has_avx512f_ = has_avx512 && (data[1] & (1 << 16));
        // Add check for AVX512 Skylake since tensorization GEMM need intrinsics from avx512bw/avx512dq.
        // avx512_skylake = avx512f | avx512vl | avx512cd | avx512bw | avx512dq
        has_avx512_skylake_ = has_avx512 && (data[1] & ((1 << 16) | (1 << 17) | (1 << 28) | (1 << 30) | (1 << 31)));
        is_hybrid_ = (data[3] & (1 << 15));
      }
    }
  }
#endif
}

}  // namespace onnxruntime
