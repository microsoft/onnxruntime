// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime_api.h>

#include "contrib_ops/cuda/quantization/quantized_decode_l2_prefetch.h"
#include "gtest/gtest.h"

namespace onnxruntime::test {

// Keep GoogleTest's RE2 dependency out of the NVCC translation unit.
void RunQuantizedDecodeL2PrefetchParity();

TEST(CUDA_EP_Unittest, QuantizedDecodeL2PrefetchParity) {
  int device, major, minor;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), cudaSuccess);
  ASSERT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), cudaSuccess);
  if (!contrib::cuda::ShouldPrefetchQuantizedDecodeL2(true, major, minor)) {
    GTEST_SKIP() << "L2 decode prefetch is enabled only on SM121";
  }
  EXPECT_NO_THROW(RunQuantizedDecodeL2PrefetchParity());
}

}  // namespace onnxruntime::test
