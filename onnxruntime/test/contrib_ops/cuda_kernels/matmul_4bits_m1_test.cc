// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include <cuda_runtime_api.h>

#include "gtest/gtest.h"

namespace onnxruntime::cuda::test {

std::string CompareMatMul4BitsM1DirectScales(int max_unroll, int k, int n, int pattern);

namespace {

template <int MaxUnroll>
void TestDirectScales() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }
  const auto compare = [](int k, int n, int pattern) {
    SCOPED_TRACE(::testing::Message() << "K=" << k << ", N=" << n << ", pattern=" << pattern);
    EXPECT_EQ("", CompareMatMul4BitsM1DirectScales(MaxUnroll, k, n, pattern));
  };
  for (int k : {8, 264, 1032, 3072, 4088, 4096, 4104, 8192}) {
    compare(k, 24, 0);
  }
  compare(3072, 1024, 0);
  compare(3072, 24, 1);
}

TEST(CUDA_EP_Unittest, MatMul4BitsM1DirectScalesUnroll4) {
  TestDirectScales<4>();
}

TEST(CUDA_EP_Unittest, MatMul4BitsM1DirectScalesUnroll16) {
  TestDirectScales<16>();
}

}  // namespace
}  // namespace onnxruntime::cuda::test
