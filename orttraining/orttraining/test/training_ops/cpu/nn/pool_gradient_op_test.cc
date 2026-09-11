// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace contrib {
namespace test {

using namespace onnxruntime::test;

namespace {
constexpr auto kOpsetVersion = 9;

void RunMaxPoolGradTest(const std::vector<int64_t>& dY_shape,
                        const std::vector<float>& dY_data,
                        const std::vector<int64_t>& indices_shape,
                        const std::vector<int64_t>& indices_data,
                        const std::vector<int64_t>& dX_shape,
                        const std::vector<float>& dX_expected,
                        bool expect_failure = false,
                        const std::string& expected_failure_msg = "") {
  OpTester t{"MaxPoolGrad", kOpsetVersion, kOnnxDomain};
  t.AddInput("dY", dY_shape, dY_data);
  t.AddInput("Indices", indices_shape, indices_data);
  t.AddOutput<float>("dX", dX_shape, dX_expected);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());

  if (expect_failure) {
    t.Run(OpTester::ExpectResult::kExpectFailure, expected_failure_msg,
          {}, nullptr, &execution_providers);
  } else {
    t.Run(OpTester::ExpectResult::kExpectSuccess, "",
          {}, nullptr, &execution_providers);
  }
}
}  // namespace

TEST(MaxPoolGradTest, Basic) {
  // dY shape: [1, 1, 2, 2] = 4 elements
  // dX shape: [1, 1, 4, 4] = 16 elements
  // Indices point to valid positions within [0, 16)
  RunMaxPoolGradTest(
      /*dY_shape=*/{1, 1, 2, 2},
      /*dY_data=*/{1.0f, 2.0f, 3.0f, 4.0f},
      /*indices_shape=*/{1, 1, 2, 2},
      /*indices_data=*/{0, 2, 8, 10},
      /*dX_shape=*/{1, 1, 4, 4},
      /*dX_expected=*/{1.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(MaxPoolGradTest, IndicesOutOfBoundsPositive) {
  // Index 1000 is far beyond dX size of 16
  RunMaxPoolGradTest(
      /*dY_shape=*/{1, 1, 2, 2},
      /*dY_data=*/{1.0f, 2.0f, 3.0f, 4.0f},
      /*indices_shape=*/{1, 1, 2, 2},
      /*indices_data=*/{0, 1000, 8, 10},
      /*dX_shape=*/{1, 1, 4, 4},
      /*dX_expected=*/std::vector<float>(16, 0.0f),
      /*expect_failure=*/true,
      /*expected_failure_msg=*/"out of range");
}

TEST(MaxPoolGradTest, IndicesOutOfBoundsNegative) {
  // Negative index is invalid
  RunMaxPoolGradTest(
      /*dY_shape=*/{1, 1, 2, 2},
      /*dY_data=*/{1.0f, 2.0f, 3.0f, 4.0f},
      /*indices_shape=*/{1, 1, 2, 2},
      /*indices_data=*/{0, -1, 8, 10},
      /*dX_shape=*/{1, 1, 4, 4},
      /*dX_expected=*/std::vector<float>(16, 0.0f),
      /*expect_failure=*/true,
      /*expected_failure_msg=*/"out of range");
}

TEST(MaxPoolGradTest, IndicesExactlyAtBoundary) {
  // Index 15 is the last valid position for dX size 16
  RunMaxPoolGradTest(
      /*dY_shape=*/{1, 1, 2, 2},
      /*dY_data=*/{1.0f, 2.0f, 3.0f, 4.0f},
      /*indices_shape=*/{1, 1, 2, 2},
      /*indices_data=*/{0, 5, 10, 15},
      /*dX_shape=*/{1, 1, 4, 4},
      /*dX_expected=*/{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f});
}

TEST(MaxPoolGradTest, IndicesOnePassedBoundary) {
  // Index 16 is one passed the last valid position (dX size = 16)
  RunMaxPoolGradTest(
      /*dY_shape=*/{1, 1, 2, 2},
      /*dY_data=*/{1.0f, 2.0f, 3.0f, 4.0f},
      /*indices_shape=*/{1, 1, 2, 2},
      /*indices_data=*/{0, 5, 10, 16},
      /*dX_shape=*/{1, 1, 4, 4},
      /*dX_expected=*/std::vector<float>(16, 0.0f),
      /*expect_failure=*/true,
      /*expected_failure_msg=*/"out of range");
}

}  // namespace test
}  // namespace contrib
}  // namespace onnxruntime
