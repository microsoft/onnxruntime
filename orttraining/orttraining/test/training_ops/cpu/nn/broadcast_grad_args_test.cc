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
                                  const std::vector<int64_t>* A_axes_expected,
                                  const std::vector<int64_t>* B_axes_expected,
                                  bool fail = false) {
  OpTester t{op, k_opset_version, kMSDomain};

  t.AddInput("a_shape", {static_cast<int64_t>(A_shape_tensor.size())}, A_shape_tensor);
  t.AddInput("b_shape", {static_cast<int64_t>(B_shape_tensor.size())}, B_shape_tensor);

  if (A_axes_expected)
    t.AddOutput<int64_t>("a_axes", {static_cast<int64_t>(A_axes_expected->size())}, *A_axes_expected);
  if (B_axes_expected)
    t.AddOutput<int64_t>("b_axes", {static_cast<int64_t>(B_axes_expected->size())}, *B_axes_expected);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  if (fail)
    t.Run(OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &execution_providers);
  else
    t.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace

// BroadcastGradientArgs

TEST(BroadcastGradientArgsTest, Basic) {
  std::vector<int64_t> A_axes_expected = {};
  std::vector<int64_t> B_axes_expected = {1, 0};
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {2, 16, 1024, 1024}, {1, 1, 1024, 1024},
                               &A_axes_expected, &B_axes_expected);
}

TEST(BroadcastGradientArgsTest, Basic_both_valid_op) {
  std::vector<int64_t> A_axes_expected = {2};
  std::vector<int64_t> B_axes_expected = {1, 0};
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {2, 16, 1, 1024}, {1, 1, 1024, 1024},
                               &A_axes_expected, &B_axes_expected);
}

TEST(BroadcastGradientArgsTest, Basic_no_bcast) {
  std::vector<int64_t> A_axes_expected = {};
  std::vector<int64_t> B_axes_expected = {};
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {2, 3, 4, 5}, {2, 3, 4, 5},
                               &A_axes_expected, &B_axes_expected);
}

TEST(BroadcastGradientArgsTest, Basic_B_scalar) {
  std::vector<int64_t> A_axes_expected = {};
  std::vector<int64_t> B_axes_expected = {3, 2, 1, 0};
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {2, 3, 4, 5}, {},
                               &A_axes_expected, &B_axes_expected);
}

TEST(BroadcastGradientArgsTest, Basic_B_vector) {
  std::vector<int64_t> A_axes_expected = {};
  std::vector<int64_t> B_axes_expected = {2, 1, 0};
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {2, 3, 4, 5}, {5},
                               &A_axes_expected, &B_axes_expected);
}

TEST(BroadcastGradientArgsTest, Basic_A_bcast_different_size) {
  std::vector<int64_t> A_axes_expected = {1, 0};
  std::vector<int64_t> B_axes_expected = {};
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {4, 5}, {2, 3, 4, 5},
                               &A_axes_expected, &B_axes_expected);
}

TEST(BroadcastGradientArgsTest, Basic_both_bcast_different_size) {
  std::vector<int64_t> A_axes_expected = {1, 0};
  std::vector<int64_t> B_axes_expected = {3, 2};
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {1, 4, 5}, {2, 3, 1, 1},
                               &A_axes_expected, &B_axes_expected);
}

TEST(BroadcastGradientArgsTest, Basic_both_bcast_different_size_2) {
  std::vector<int64_t> A_axes_expected = {0};
  std::vector<int64_t> B_axes_expected = {3, 2, 1};
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {3, 4, 5}, {2, 1, 1, 1},
                               &A_axes_expected, &B_axes_expected);
}

TEST(BroadcastGradientArgsTest, Basic_invalid_broadcast) {
  std::vector<int64_t> A_axes_expected = {};
  std::vector<int64_t> B_axes_expected = {};
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {3, 4, 5}, {2, 1, 6, 1},
                               &A_axes_expected, &B_axes_expected, true /*fail*/);
}

TEST(BroadcastGradientArgsTest, Basic_only_A_output) {
  std::vector<int64_t> A_axes_expected = {0};
  RunBroadcastGradientArgsTest("BroadcastGradientArgs", {3, 4, 5}, {2, 1, 1, 1},
                               &A_axes_expected, nullptr);
}

}  // namespace test
}  // namespace contrib
}  // namespace onnxruntime
