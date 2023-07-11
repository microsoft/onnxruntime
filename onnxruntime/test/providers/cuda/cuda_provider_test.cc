// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#include "core/common/status.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#include "gtest/gtest.h"
namespace onnxruntime {

ProviderInfo_CUDA& GetProviderInfo_CUDA_Test();

namespace test {
namespace cuda {
TEST(CUDA_EP_Unittest, All) {
  onnxruntime::ProviderInfo_CUDA& ep = onnxruntime::GetProviderInfo_CUDA_Test();
  ep.TestAll();
}

}  // namespace cuda
}  // namespace test
}  // namespace onnxruntime
