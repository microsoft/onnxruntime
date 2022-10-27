// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "test/common/dnnl_op_test_utils.h"
#include "core/common/cpuid_info.h"
#include "core/platform/env.h"
#include <mutex>

namespace onnxruntime {
namespace test {
bool DnnlSupportedGpuFound() {
// This assumes that if the code was built using the DNNL_GPU_RUNTIME then you have GPU support.
// It is possible this is not true.
// If a compatable GPU is not found at runtime the oneDNN ep will run the bfloat16 code on the CPU.
// If there is no GPU or CPU support for bfloat16 this assumption may cause unit tests to fail.
// They will fail with a "Could not find an implementation for [operator]" error.
#if defined(DNNL_GPU_RUNTIME)
 return true;
#else
 return false;
#endif
}
std::once_flag once_flag1;

bool DnnlHasBF16Support() {
  if (DnnlSupportedGpuFound()){
    return true;
  }
  // HasAVX512Skylake checks for AVX512BW which can run bfloat16 but
  // is slower than float32 by 3x to 4x.
  static bool use_all_bf16_hardware = false;
  std::call_once(once_flag1, []() {
    const std::string bf16_env = Env::Default().GetEnvironmentVar("ORT_DNNL_USE_ALL_BF16_HW");
    if (!bf16_env.empty()) {
      use_all_bf16_hardware = (std::stoi(bf16_env) == 0 ? false : true);
    }
  });

  // HasAVX512Skylake checks for AVX512BW which can run bfloat16 but
  // is slower than float32 by 3x to 4x.
  // By default the AVX512BW ISA is not used. It is still useful for validation
  // so it can be enabled by setting the environment variable ORT_DNNL_USE_ALL_BF16_HW=1
  if (use_all_bf16_hardware && CPUIDInfo::GetCPUIDInfo().HasAVX512Skylake()) {
    return true;
  }

  // If AVX512-BF16 or AMX-BF16 exist then bfloat16 ops are HW accelerated
  if (CPUIDInfo::GetCPUIDInfo().HasAVX512_BF16() ||
      CPUIDInfo::GetCPUIDInfo().HasAMX_BF16()) {
    return true;
  }
  return false;
}
}
}
