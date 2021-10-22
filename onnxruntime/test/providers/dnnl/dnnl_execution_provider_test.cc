// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
TEST(DNNLExecutionProviderTest, MetadataTest) {
    #if 0 // With DNNL as a DLL this can't be tested here TODO(pranav)
  DNNLExecutionProviderInfo info;
  info.create_arena = false;
  auto provider = std::make_unique<DNNLExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);
  ASSERT_STREQ(provider->GetAllocator(0, OrtMemTypeCPUOutput)->Info().name, "DnnlCpu");
  #endif
}
}  // namespace test
}  // namespace onnxruntime
