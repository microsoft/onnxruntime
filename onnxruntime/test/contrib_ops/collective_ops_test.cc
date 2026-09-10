// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#ifdef ORT_USE_NCCL
namespace {

void RunInvalidAllGatherTest(const std::vector<int64_t>& input_shape,
                             int64_t axis,
                             int64_t group_size,
                             const std::string& expected_error) {
  OpTester test("AllGather", 1, kMSDomain);
  test.AddAttribute("axis", axis);
  test.AddAttribute("group_size", group_size);
  const std::vector<float> input_data = input_shape.empty() ? std::vector<float>{0.0f} : std::vector<float>{};
  test.AddInput<float>("input", input_shape, input_data);
  test.AddOutput<float>("output", {0}, {});
  test.Run(OpTester::ExpectResult::kExpectFailure, expected_error);
}

}  // namespace

TEST(CollectiveOpsTest, AllGatherRejectsInvalidAxis) {
  RunInvalidAllGatherTest({}, 0, 1, "axis must be in the range [0, 0)");
  RunInvalidAllGatherTest({0, 1}, -1, 1, "axis must be in the range [0, 2)");
  RunInvalidAllGatherTest({0, 1}, 2, 1, "axis must be in the range [0, 2)");
}

TEST(CollectiveOpsTest, AllGatherRejectsInvalidGroupSize) {
  RunInvalidAllGatherTest({0, 1}, 0, 0, "group_size must be greater than 0");
}

TEST(CollectiveOpsTest, AllGatherRejectsOutputDimensionOverflow) {
  RunInvalidAllGatherTest({0, std::numeric_limits<int64_t>::max()}, 1, 2,
                          "AllGather output dimension is too large");
}
#endif

}  // namespace test
}  // namespace onnxruntime