// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/mlas/inc/mlas.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

TEST(NchwcOpsTest, ReorderOutputRejectsUnalignedInputChannels) {
  const int64_t block_size = static_cast<int64_t>(MlasNchwcGetBlockSize());
  if (block_size <= 1) {
    GTEST_SKIP() << "NCHWc blocking is not enabled on this platform.";
  }

  const int64_t input_channels = block_size - 1;
  OpTester test("ReorderOutput", 1, kMSNchwcDomain);
  test.AddAttribute("channels", int64_t{1});
  test.AddAttribute("channels_last", int64_t{0});
  test.AddInput<float>("X", {1, input_channels, 2, 2},
                       std::vector<float>(static_cast<size_t>(input_channels) * 4, 0.0f));
  test.AddOutput<float>("Y", {1, 1, 2, 2}, {0.0f, 0.0f, 0.0f, 0.0f});

  test.Config(OpTester::ExpectResult::kExpectFailure,
              "Input channels must match the NCHWc block-aligned channel count.")
      .ConfigEp(DefaultCpuExecutionProvider())
      .RunWithConfig();
}

}  // namespace test
}  // namespace onnxruntime