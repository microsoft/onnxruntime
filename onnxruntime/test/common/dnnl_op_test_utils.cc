// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "test/common/dnnl_op_test_utils.h"
#include "core/common/cpuid_info.h"

namespace onnxruntime {
namespace test {
bool DnnlSupportedGpuFound() {
#if defined(DNNL_GPU_RUNTIME)
 return true;
#else
 return false;
#endif
}

bool DnnlHasBF16Support() {
  if (DnnlSupportedGpuFound()){
    return true;
  }
  // HasAVX512Skylake checks for AVX512BW which can run bfloat16 but
  // is slower than float32 by 3x to 4x.
  // If AVX512-BF16 or AMX-BF16 exist then bfloat16 ops are HW accelerated
  if (CPUIDInfo::GetCPUIDInfo().HasAVX512Skylake() ||
      CPUIDInfo::GetCPUIDInfo().HasAVX512_BF16() ||
      CPUIDInfo::GetCPUIDInfo().HasAMX_BF16()) {
    return true;
  }
  return false;
}
}
}
