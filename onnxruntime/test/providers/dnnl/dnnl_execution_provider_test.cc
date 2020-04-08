// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
TEST(DNNLExecutionProviderTest, MetadataTest) {
  DNNLExecutionProviderInfo info;
  info.create_arena = false;
  auto provider = onnxruntime::make_unique<DNNLExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);
  ASSERT_STREQ(provider->GetAllocator(0, OrtMemTypeCPUOutput)->Info().name, "DnnlCpu");
}
}  // namespace test
}  // namespace onnxruntime
