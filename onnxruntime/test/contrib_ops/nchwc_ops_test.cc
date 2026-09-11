// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/mlas/inc/mlas.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

void RunInvalidNchwcConvTest(const std::vector<int64_t>& input_shape,
                             const std::vector<int64_t>& filter_shape,
                             const std::vector<int64_t>* bias_shape,
                             int64_t group,
                             const std::string& expected_error) {
  OpTester test("Conv", 1, kMSNchwcDomain);
  test.AddAttribute("group", group);
  test.AddInput<float>("X", input_shape, {});
  test.AddInput<float>("W", filter_shape,
                       std::vector<float>(static_cast<size_t>(TensorShape(filter_shape).Size()), 0.0f));
  if (bias_shape != nullptr) {
    test.AddInput<float>("B", *bias_shape,
                         std::vector<float>(static_cast<size_t>(TensorShape(*bias_shape).Size()), 0.0f));
  }
  test.AddOutput<float>("Y", {0, filter_shape[0], 1, 1}, {});

  test.Config(OpTester::ExpectResult::kExpectFailure, expected_error)
      .ConfigEp(DefaultCpuExecutionProvider())
      .RunWithConfig();
}

}  // namespace

TEST(NchwcOpsTest, ConvRejectsUnalignedOutputChannels) {
  const int64_t block_size = static_cast<int64_t>(MlasNchwcGetBlockSize());
  if (block_size <= 1) {
    GTEST_SKIP() << "NCHWc blocking is not enabled on this platform.";
  }

  RunInvalidNchwcConvTest({0, 1, 1, 1}, {block_size - 1, 1, 1, 1}, nullptr, 1,
                          "NCHWc Conv input and filter shapes do not match a supported blocked layout.");
}

TEST(NchwcOpsTest, ConvRejectsInvalidGroup) {
  const int64_t block_size = static_cast<int64_t>(MlasNchwcGetBlockSize());
  if (block_size <= 1) {
    GTEST_SKIP() << "NCHWc blocking is not enabled on this platform.";
  }

  RunInvalidNchwcConvTest({0, 1, 1, 1}, {block_size, 1, 1, 1}, nullptr, 0,
                          "NCHWc Conv group must be greater than 0.");
}

TEST(NchwcOpsTest, ConvRejectsMismatchedBias) {
  const int64_t block_size = static_cast<int64_t>(MlasNchwcGetBlockSize());
  if (block_size <= 1) {
    GTEST_SKIP() << "NCHWc blocking is not enabled on this platform.";
  }

  const std::vector<int64_t> bias_shape{block_size - 1};
  RunInvalidNchwcConvTest({0, 1, 1, 1}, {block_size, 1, 1, 1}, &bias_shape, 1,
                          "NCHWc Conv bias must be a 1D tensor matching the physical output channels.");
}

TEST(NchwcOpsTest, ConvRejectsNonVectorBias) {
  const int64_t block_size = static_cast<int64_t>(MlasNchwcGetBlockSize());
  if (block_size <= 1) {
    GTEST_SKIP() << "NCHWc blocking is not enabled on this platform.";
  }

  const std::vector<int64_t> bias_shape{1, block_size};
  RunInvalidNchwcConvTest({0, 1, 1, 1}, {block_size, 1, 1, 1}, &bias_shape, 1,
                          "NCHWc Conv bias must be a 1D tensor matching the physical output channels.");
}

TEST(NchwcOpsTest, ConvRejectsUnsupportedGroupedLayout) {
  const int64_t block_size = static_cast<int64_t>(MlasNchwcGetBlockSize());
  if (block_size <= 1) {
    GTEST_SKIP() << "NCHWc blocking is not enabled on this platform.";
  }

  const int64_t input_channels_per_group = block_size / 2;
  RunInvalidNchwcConvTest({0, input_channels_per_group * 2, 1, 1},
                          {block_size * 2, input_channels_per_group, 1, 1}, nullptr, 2,
                          "NCHWc Conv input and filter shapes do not match a supported blocked layout.");
}

TEST(NchwcOpsTest, ConvRejectsUnalignedOutputChannelsPerGroup) {
  const int64_t block_size = static_cast<int64_t>(MlasNchwcGetBlockSize());
  if (block_size <= 1) {
    GTEST_SKIP() << "NCHWc blocking is not enabled on this platform.";
  }

  RunInvalidNchwcConvTest({0, block_size * 2, 1, 1}, {block_size, block_size, 1, 1}, nullptr, 2,
                          "NCHWc Conv input and filter shapes do not match a supported blocked layout.");
}

}  // namespace test
}  // namespace onnxruntime