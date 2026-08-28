// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <limits>
#include <vector>

#include "core/session/onnxruntime_session_options_config_keys.h"
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
  NonZeroBasicNumericTest<uint8_t>();
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
  // TODO: ONNX shape inference disagrees about the output shape.
  // ONNX spec is ambiguous: https://github.com/onnx/onnx/issues/2428.
  // Once spec clarified, remove strict_shape_type_inference override.
  SessionOptions so;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsConfigStrictShapeTypeInference, "0"));
  {
    OpTester test{kOpName, kOpVersion};
    test.AddInput<int32_t>("X", {}, {0});
#ifdef USE_TENSORRT
    // TensorRT follows ONNX spec where NonZero produces output shape (0, N) instead of (1, N) for scalar input
    test.AddOutput<int64_t>("Y", {0, 0}, {});
#else
    test.AddOutput<int64_t>("Y", {1, 0}, {});
#endif
    test.Run(so);
  }
  {
    OpTester test{kOpName, kOpVersion};
    test.AddInput<int32_t>("X", {}, {1});
#ifdef USE_TENSORRT
    // TensorRT follows ONNX spec where NonZero produces output shape (0, N) instead of (1, N) for scalar input
    test.AddOutput<int64_t>("Y", {0, 1}, {});
#else
    test.AddOutput<int64_t>("Y", {1, 1}, {0});
#endif
    test.Run(so);
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

TEST(NonZeroOpTest, OneDimensional) {
  OpTester test{kOpName, kOpVersion};
  test.AddInput<int32_t>("X", {5}, {0, 3, 0, 0, 7});
  test.AddOutput<int64_t>("Y", {1, 2}, {1, 4});
  test.Run();
}

TEST(NonZeroOpTest, EmptyOneDimensional) {
  OpTester test{kOpName, kOpVersion};
  test.AddInput<int32_t>("X", {0}, {});
  test.AddOutput<int64_t>("Y", {1, 0}, {});
  test.Run();
}

TEST(NonZeroOpTest, AllZeros) {
  OpTester test{kOpName, kOpVersion};
  test.AddInput<int32_t>("X", {2, 3}, {0, 0, 0, 0, 0, 0});
  test.AddOutput<int64_t>("Y", {2, 0}, {});
  test.Run();
}

TEST(NonZeroOpTest, AllNonZero) {
  OpTester test{kOpName, kOpVersion};
  test.AddInput<int32_t>("X", {2, 2}, {1, 2, 3, 4});
  test.AddOutput<int64_t>(
      "Y", {2, 4},
      {0, 0, 1, 1,
       0, 1, 0, 1});
  test.Run();
}

// A value is selected when it compares unequal to zero. That includes NaN, and excludes
// negative zero.
TEST(NonZeroOpTest, FloatNaNAndNegativeZero) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  OpTester test{kOpName, kOpVersion};
  test.AddInput<float>(
      "X", {2, 3},
      {0.0f, -0.0f, nan,
       -1.5f, 0.0f, 2.5f});
  test.AddOutput<int64_t>(
      "Y", {2, 3},
      {0, 1, 1,
       2, 0, 2});
  test.Run();
}

// Ranks above three take a generic coordinate-decomposition path in the kernel.
TEST(NonZeroOpTest, FiveDims) {
  std::vector<int32_t> X(32, 0);
  X[0] = 1;   // 0,0,0,0,0
  X[9] = 1;   // 0,1,0,0,1
  X[22] = 1;  // 1,0,1,1,0
  X[31] = 1;  // 1,1,1,1,1

  OpTester test{kOpName, kOpVersion};
  test.AddInput<int32_t>("X", {2, 2, 2, 2, 2}, X);
  test.AddOutput<int64_t>(
      "Y", {5, 4},
      {0, 0, 1, 1,
       0, 1, 0, 1,
       0, 0, 1, 1,
       0, 0, 1, 1,
       0, 1, 0, 1});
  test.Run();
}

// The kernel only splits the scan across threads above an internal element-count
// threshold, so this input is deliberately much larger than the cases above. The
// non-zeros use an irregular stride so they do not align with shard boundaries.
TEST(NonZeroOpTest, LargeInput) {
  constexpr int64_t kRows = 64;
  constexpr int64_t kCols = 512;
  constexpr int64_t kTotal = kRows * kCols;

  std::vector<int32_t> X(static_cast<size_t>(kTotal), 0);
  std::vector<int64_t> expected_rows;
  std::vector<int64_t> expected_cols;
  for (int64_t i = 0; i < kTotal; ++i) {
    // Include the very last element so the final shard ends on a selected value.
    if (i % 97 == 0 || i == kTotal - 1) {
      X[static_cast<size_t>(i)] = 5;
      expected_rows.push_back(i / kCols);
      expected_cols.push_back(i % kCols);
    }
  }

  std::vector<int64_t> expected;
  expected.reserve(expected_rows.size() + expected_cols.size());
  expected.insert(expected.end(), expected_rows.begin(), expected_rows.end());
  expected.insert(expected.end(), expected_cols.begin(), expected_cols.end());

  OpTester test{kOpName, kOpVersion};
  test.AddInput<int32_t>("X", {kRows, kCols}, X);
  test.AddOutput<int64_t>("Y", {2, static_cast<int64_t>(expected_rows.size())}, expected);
  test.Run();
}

// Same size as LargeInput but fully dense, so every shard is saturated.
// std::vector<bool> cannot be handed to OpTester (no contiguous storage), so this uses
// the byte-sized mask type instead.
TEST(NonZeroOpTest, LargeInputAllNonZero) {
  constexpr int64_t kRows = 128;
  constexpr int64_t kCols = 256;
  constexpr int64_t kTotal = kRows * kCols;

  std::vector<uint8_t> X(static_cast<size_t>(kTotal), 1);
  std::vector<int64_t> expected;
  expected.reserve(static_cast<size_t>(2 * kTotal));
  for (int64_t i = 0; i < kTotal; ++i) {
    expected.push_back(i / kCols);
  }
  for (int64_t i = 0; i < kTotal; ++i) {
    expected.push_back(i % kCols);
  }

  OpTester test{kOpName, kOpVersion};
  test.AddInput<uint8_t>("X", {kRows, kCols}, X);
  test.AddOutput<int64_t>("Y", {2, kTotal}, expected);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
