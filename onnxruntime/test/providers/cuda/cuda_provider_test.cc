// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef NDEBUG
#include "gtest/gtest.h"
#include "core/common/status.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#include <iostream>
namespace onnxruntime {

ProviderInfo_CUDA& GetProviderInfo_CUDA();

namespace test {
namespace cuda {
TEST(CUDAEPTEST, ALL) {
  onnxruntime::ProviderInfo_CUDA& ep = onnxruntime::GetProviderInfo_CUDA();
  ASSERT_TRUE(ep.TestAll());
}

}  // namespace cuda
}  // namespace test
}  // namespace onnxruntime

#endif
