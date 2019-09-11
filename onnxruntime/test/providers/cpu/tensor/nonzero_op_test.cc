// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {
constexpr char kOpName[] = "NonZero";
constexpr int kOpVersion = 9;

template <typename TTarget, typename TNarrow = int8_t>
void NonZeroBasicNumericTest() {
  OpTester test{kOpName, kOpVersion};

  std::vector<int64_t> X_dims{1, 2, 3};
  std::vector<TNarrow> X{0, 1, 2,
                         0, 3, 4};
  test.AddInput<TTarget>("X", X_dims, std::vector<TTarget>{X.begin(), X.end()});
  test.AddOutput<int64_t>(
      "Y", {3, 4},
      {0, 0, 0, 0,
       0, 0, 1, 1,
       1, 2, 1, 2});
  test.Run();
}
}  // namespace

TEST(NonZeroOpTest, BasicNumeric) {
  NonZeroBasicNumericTest<int32_t>();
  NonZeroBasicNumericTest<int64_t>();
  NonZeroBasicNumericTest<float>();
}

TEST(NonZeroOpTest, BasicBool) {
  OpTester test{kOpName, kOpVersion};
  test.AddInput<bool>(
      "X", {2, 3},
      {true, false, false,
       false, false, true});
  test.AddOutput<int64_t>(
      "Y", {2, 2},
      {0, 1,
       0, 2});
  test.Run();
}

TEST(NonZeroOpTest, ThreeDims) {
  OpTester test{kOpName, kOpVersion};

  std::vector<int64_t> X_dims{2, 2, 2};
  std::vector<int64_t> X{0, 1,
                         1, 0,

                         1, 0,
                         1, 0};
  test.AddInput<int64_t>("X", X_dims, std::vector<int64_t>{X.begin(), X.end()});
  test.AddOutput<int64_t>(
      "Y", {3, 4},
      {0, 0, 1, 1,
       0, 1, 0, 1,
       1, 0, 0, 0});

  test.Run();
}

TEST(NonZeroOpTest, Scalar) {
  {
    OpTester test{kOpName, kOpVersion};
    test.AddInput<int32_t>("X", {}, {0});
    test.AddOutput<int64_t>("Y", {1, 0}, {});
    test.Run();
  }
  {
    OpTester test{kOpName, kOpVersion};
    test.AddInput<int32_t>("X", {}, {1});
    test.AddOutput<int64_t>("Y", {1, 1}, {0});
    test.Run();
  }
}

TEST(NonZeroOpTest, EmptyInput) {
  OpTester test{kOpName, kOpVersion};
  test.AddInput<int32_t>(
      "X", {1, 0, 2},
      {});
  test.AddOutput<int64_t>(
      "Y", {3, 0},
      {});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
