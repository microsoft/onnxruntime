// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Microsoft C/C++-compatible compiler
#if defined(_MSC_VER)
#include <intrin.h>
// GCC-compatible compiler targeting x86/x86-64
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#include <x86intrin.h>
#endif

#include <mutex>

#include "core/common/denormal.h"

namespace onnxruntime {

#ifdef _OPENMP
class DenormalAsZeroInitializer {
 public:
  explicit DenormalAsZeroInitializer(bool on) : on_(on) {}

  DenormalAsZeroInitializer(const DenormalAsZeroInitializer& init) : on_(init.on_) {
    SetDenormalAsZero(on_);
  }

 private:
  bool on_;
};

void InitializeWithDenormalAsZero(bool on) {
  DenormalAsZeroInitializer init(on);
#pragma omp parallel for firstprivate(init)
  for (auto i = 0; i < 1; ++i) {
  }
}
#endif

bool SetDenormalAsZero(bool on) {
#if defined(__SSE3__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
  static std::once_flag once;
  static volatile bool has_sse3 = true;
  std::call_once(once, [&] {
#if defined(_M_AMD64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
    unsigned cpuid[4];
#if defined(_WIN32)
    __cpuid((int*)cpuid, 1);
#else
        __cpuid(1, cpuid[0], cpuid[1], cpuid[2], cpuid[3]);
#endif
    has_sse3 = (cpuid[2] & 0x1) == 0x1;
#else
        has_sse3 = false;
#endif
  });

  // runtime check for denormal-as-zero support.
  if (has_sse3) {
    if (on) {
      _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    } else {
      _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    }
    return true;
  }
#endif
  return false;
}

}  // namespace onnxruntime
