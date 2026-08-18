// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GPU-free coverage for the SM90 (Hopper) weight_prepacked=2 validation logic used by
// contrib_ops/cuda/quantization/matmul_nbits.h's MatMulNBits<T> constructor.
//
// MatMulNBits<T> normally reads the device compute capability from real hardware
// (GetDeviceProp()), so the "native SM90 kernel is not compiled in this build" throw (e.g. on
// Windows/MSVC, see the COMPILE_HOPPER_TMA_GEMMS macro in cmake/onnxruntime_providers_cuda.cmake) is
// only reachable on an actual Hopper GPU that was built without the native kernel -- a combination
// existing tests cannot produce, since they all GTEST_SKIP() when no CUDA device is present (see
// e.g. MatMulNBits.Fp16_Int4_PrepackedSm90BlockSize32Rejected in test/contrib_ops/matmul_4bits_test.cc).
// ValidateSm90PrepackedWeightSupport() takes sm/block_size as plain parameters instead of reading
// real hardware, so it can be exercised here with synthetic values and no GPU at all.
//
// This lives alongside the other CUDA-EP-internal tests under contrib_ops/cuda_kernels/ (rather than
// in test/contrib_ops/matmul_4bits_test.cc) because onnxruntime_providers_cuda is a runtime-loaded
// shared library (see core/providers/cuda/symbols.def, which exports only the small provider-bridge
// interface): matmul_nbits.cc's free functions are not exported from that DLL/so, so a normal
// provider-test binary cannot call them directly (it never links onnxruntime_providers_cuda at
// compile time -- see AddTest()/DEPENDS in cmake/onnxruntime_unittests.cmake). Files under
// contrib_ops/cuda_kernels/ are instead compiled directly into the onnxruntime_providers_cuda_ut
// module together with the CUDA EP's own object files whenever
// onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS is set (as it is in the windows_cuda.yml / linux_cuda_ci.yml
// CI legs), so this test both links successfully and observes COMPILE_HOPPER_TMA_GEMMS exactly as the
// real onnxruntime_providers_cuda target does -- no ODR/macro-skew risk.
//
// Test can be run like the following:
//  ./onnxruntime_provider_test --gtest_filter=CUDA_EP_Unittest.*
#if USE_FPA_INTB_GEMM
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <string>

#include "core/common/common.h"
#include "contrib_ops/cuda/quantization/matmul_nbits_sm90_validation.h"

namespace onnxruntime {
namespace test {

namespace {
// Runs `fn` and returns the message of any thrown exception, or "" if it did not throw.
template <typename Fn>
std::string CaughtMessage(Fn&& fn) {
  std::string message;
  ORT_TRY {
    fn();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() { message = ex.what(); });
  }
  return message;
}
}  // namespace

// Non-Hopper compute capability is always rejected, regardless of whether this build compiled the
// native SM90 kernel. This assertion is build-independent.
TEST(MatMulNBitsSm90ValidationTest, RejectsNonHopperComputeCapability) {
  const std::string message = CaughtMessage([]() {
    onnxruntime::contrib::cuda::ValidateSm90PrepackedWeightSupport(/*sm=*/80, /*block_size=*/64);
  });
  EXPECT_THAT(message, ::testing::HasSubstr("weight_prepacked=2 (SM90 layout) requires a compute capability 9.0"));
}

// A Hopper (sm=90) request is validated differently depending on whether this build actually
// compiled the native SM90 (Hopper TMA/WGMMA) fpA_intB kernel.
TEST(MatMulNBitsSm90ValidationTest, Sm90SupportMatchesBuildCapability) {
  using onnxruntime::contrib::cuda::IsNativeSm90FpAIntBGemmCompiled;
  using onnxruntime::contrib::cuda::ValidateSm90PrepackedWeightSupport;

  if (IsNativeSm90FpAIntBGemmCompiled()) {
    // The native SM90 kernel is available: a supported block_size (64 or 128) must be accepted ...
    EXPECT_NO_THROW(ValidateSm90PrepackedWeightSupport(/*sm=*/90, /*block_size=*/64));

    // ... but block_size=32 is SM80/GEMV-only and must still be rejected.
    const std::string message = CaughtMessage([]() {
      ValidateSm90PrepackedWeightSupport(/*sm=*/90, /*block_size=*/32);
    });
    EXPECT_THAT(message, ::testing::HasSubstr("supports block_size 64 or 128 only"));
  } else {
    // The native SM90 kernel is not compiled in this build (e.g. Windows/MSVC): even a Hopper
    // device with an otherwise-supported block_size must be rejected with a build-support message.
    const std::string message = CaughtMessage([]() {
      ValidateSm90PrepackedWeightSupport(/*sm=*/90, /*block_size=*/64);
    });
    EXPECT_THAT(message, ::testing::HasSubstr("is not supported by this ONNX Runtime build"));
  }
}

}  // namespace test
}  // namespace onnxruntime
#endif  // USE_FPA_INTB_GEMM
