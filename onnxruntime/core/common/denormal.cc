// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Microsoft C/C++-compatible compiler
#if defined(_MSC_VER)
#include <intrin.h>
// GCC-compatible compiler targeting x86/x86-64
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#include <cpuinfo.h>
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
// Compile if we have SSE support (GCC), x86-64 (MSVC), or x86 with SSE (MSVC)
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
  static std::once_flag once;
  std::call_once(once, cpuinfo_initialize);

  if (cpuinfo_has_x86_daz()) {
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
