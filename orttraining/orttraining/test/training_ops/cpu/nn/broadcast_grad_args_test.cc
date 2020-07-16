// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>

#include "gtest/gtest.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace contrib {
namespace test {

using namespace onnxruntime::test;

namespace {
constexpr auto k_opset_version = 1;

void RunBroadcastGradientArgsTest(const char* op,
                                  const std::vector<int64_t>& A_shape_tensor,
                                  const std::vector<int64_t>& B_shape_tensor,
                                  const std::vector<int64_t>& A_axes_expected,
                                  const std::vector<int64_t>& B_axes_expected,
                                  bool fail = false) {
  OpTester t{op, k_opset_version, kMSDomain};

  t.AddInput("a_shape", {static_cast<int64_t>(A_shape_tensor.size())}, A_shape_tensor);
  t.AddInput("b_shape", {static_cast<int64_t>(B_shape_tensor.size())}, B_shape_tensor);

  t.AddOutput<int64_t>("a_axes", {static_cast<int64_t>(A_axes_expected.size())}, A_axes_expected);
  t.AddOutput<int64_t>("b_axes", {static_cast<int64_t>(B_axes_expected.size())}, B_axes_expected);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  if (fail)
    t.Run(OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &execution_providers, ExecutionMode::ORT_SEQUENTIAL);
  else
    t.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers, ExecutionMode::ORT_SEQUENTIAL);
}

}  // namespace

// BroadcastGradientArgs

TEST(BroadcastGradientArgsTest, Basic) {
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {2, 16, 1024, 1024}, {1, 1, 1024, 1024},
                               {}, {1, 0});
}

TEST(BroadcastGradientArgsTest, Basic_both_valid_op) {
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {2, 16, 1, 1024}, {1, 1, 1024, 1024},
                               {2}, {1, 0});
}

TEST(BroadcastGradientArgsTest, Basic_no_bcast) {
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {2, 3, 4, 5}, {2, 3, 4, 5},
                               {}, {});
}

TEST(BroadcastGradientArgsTest, Basic_B_scalar) {
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {2, 3, 4, 5}, {},
                               {}, {3, 2, 1, 0});
}

TEST(BroadcastGradientArgsTest, Basic_B_vector) {
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {2, 3, 4, 5}, {5},
                               {}, {2, 1, 0});
}

TEST(BroadcastGradientArgsTest, Basic_A_bcast_different_size) {
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {4, 5}, {2, 3, 4, 5},
                               {1, 0}, {});
}

TEST(BroadcastGradientArgsTest, Basic_both_bcast_different_size) {
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {1, 4, 5}, {2, 3, 1, 1},
                               {1, 0}, {3, 2});
}

TEST(BroadcastGradientArgsTest, Basic_both_bcast_different_size_2) {
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {3, 4, 5}, {2, 1, 1, 1},
                               {0}, {3, 2, 1});
}

TEST(BroadcastGradientArgsTest, Basic_invalid_broadcast) {
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {3, 4, 5}, {2, 1, 6, 1},
                               {}, {}, true /*fail*/);
}

}  // namespace test
}  // namespace contrib
}  // namespace onnxruntime
