// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <type_traits>
#include <memory>
#include <utility>

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// Combinations: types, gather_axis, quantize_axis, block_size, indices, scale shape vs data shape
template <typename T1, typename T2, typename Tind>
void RunGatherBlockQuantized(const std::vector<T1>& data,
                             const std::vector<int64_t>& data_shape,
                             const std::vector<Tind>& indices,
                             const std::vector<int64_t>& indices_shape,
                             const std::vector<T2>& scales,
                             const std::vector<int64_t>& scales_shape,
                             const std::vector<T1>& zero_points,
                             const int64_t gather_axis,
                             const int64_t quantize_axis,
                             const int64_t block_size,
                             const std::vector<T2>& output,
                             const std::vector<int64_t>& output_shape,
                             OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess) {
  auto run_test = [&](bool indices_is_initializer) {
    OpTester test("GatherBlockQuantized", 1, kMSDomain);

    test.AddAttribute<int64_t>("gather_axis", gather_axis);
    test.AddAttribute<int64_t>("quantize_axis", quantize_axis);
    test.AddAttribute<int64_t>("block_size", block_size);

    test.AddInput<T1>("data", data_shape, data);
    test.AddInput<Tind>("indices", indices_shape, indices, indices_is_initializer);
    test.AddInput<T2>("scales", scales_shape, scales);
    if (!zero_points.empty()) {
      test.AddInput<T1>("zero_points", scales_shape, zero_points);
    }

    test.AddOutput<T2>("output", output_shape, output);

    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCpuExecutionProvider());
    test.Run(expect_result, "", {}, nullptr, &eps);
  };

  run_test(false);
  run_test(true);
}

template <typename T1, typename T2>
typename std::enable_if<
    (boost::mp11::mp_contains<TypeList<BFloat16, MLFloat16, float>, T1>::value && std::is_same<T2, float>::value) ||
        (std::is_integral<T1>::value && std::is_same<T2, int>::value),
    std::vector<T1>>::type
ToType(const std::vector<T2>& vec) {
  std::vector<T1> result;
  for (auto v : vec) {
    result.push_back(static_cast<T1>(v));
  }

  return result;
}

template <typename T>
typename std::enable_if<boost::mp11::mp_contains<TypeList<UInt4x2, Int4x2>, T>::value, std::vector<T>>::type
ToType(const std::vector<int>& vec) {
  std::vector<T> result;
  size_t i = 0;
  constexpr int offset = std::is_same<T, Int4x2>::value ? 0 : 8;
  for (i = 0; i + 1 < vec.size(); i += 2) {
    result.push_back(T(vec[i] + offset, vec[i + 1] + offset));
  }
  if (i < vec.size()) {
    result.push_back(T(vec[i] + offset, 0 + offset));
  }

  return result;
}

template <typename T1, typename T2, typename Tind>
void Test_Fail_WithZeroPoints(int64_t gather_axis,
                              int64_t quantize_axis,
                              int64_t block_size) {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};
  std::vector<int> zero_points = {-1, 1, 0, 0, 1, -1};
  std::vector<float> output = {8.f, 10.f, 12.f, 14.f,
                               3.f, 4.f, 5.f, 6.f,
                               -6.f, -4.f, -2.f, 0.f};
  std::vector<int64_t> output_shape = {1, 3, 4};

  RunGatherBlockQuantized(ToType<T1>(data),
                          data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          ToType<T1>(zero_points),
                          gather_axis,
                          quantize_axis,
                          block_size,
                          ToType<T2>(output),
                          output_shape,
                          OpTester::ExpectResult::kExpectFailure);
}

TEST(GatherBlockQuantizedOpTest, UnsupportedTypes) {
  Test_Fail_WithZeroPoints<int8_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<uint8_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<int16_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<uint16_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<int32_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<uint32_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<int64_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<uint64_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<UInt4x2, float, int16_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<Int4x2, float, int16_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<UInt4x2, BFloat16, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<Int4x2, BFloat16, int32_t>(0, 2, 16);
}

TEST(GatherBlockQuantizedOpTest, InvalidBlockSize) {
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 8);
  Test_Fail_WithZeroPoints<Int4x2, float, int32_t>(0, 2, 17);
}

TEST(GatherBlockQuantizedOpTest, InvalidGatherAxis) {
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(3, 2, 16);
  Test_Fail_WithZeroPoints<Int4x2, float, int32_t>(-4, 2, 16);
}

TEST(GatherBlockQuantizedOpTest, InvalidQuantizeAxis) {
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 3, 16);
  Test_Fail_WithZeroPoints<Int4x2, float, int32_t>(0, -4, 16);
}

template <typename T1, typename T2, typename Tind>
void Test_ShapeMismatch_WithZeroPoints() {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 2, 1};
  std::vector<int> zero_points = {-1, 1, 0, 0};
  std::vector<float> output = {8.f, 10.f, 12.f, 14.f,
                               3.f, 4.f, 5.f, 6.f,
                               -6.f, -4.f, -2.f, 0.f};
  std::vector<int64_t> output_shape = {1, 3, 4};

  RunGatherBlockQuantized(ToType<T1>(data),
                          data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          ToType<T1>(zero_points),
                          0,
                          2,
                          16,
                          ToType<T2>(output),
                          output_shape,
                          OpTester::ExpectResult::kExpectFailure);
}

TEST(GatherBlockQuantizedOpTest, ShapeMismatch) {
  Test_ShapeMismatch_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_ShapeMismatch_WithZeroPoints<Int4x2, float, int32_t>();
}

template <typename T1, typename T2, typename Tind>
void Test_InvalidIndices_WithZeroPoints() {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  std::vector<int> indices = {2};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};
  std::vector<int> zero_points = {-1, 1, 0, 0, 1, -1};
  std::vector<float> output = {8.f, 10.f, 12.f, 14.f,
                               3.f, 4.f, 5.f, 6.f,
                               -6.f, -4.f, -2.f, 0.f};
  std::vector<int64_t> output_shape = {1, 3, 4};

  RunGatherBlockQuantized(ToType<T1>(data),
                          data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          ToType<T1>(zero_points),
                          0,
                          2,
                          16,
                          ToType<T2>(output),
                          output_shape,
                          OpTester::ExpectResult::kExpectFailure);
}

TEST(GatherBlockQuantizedOpTest, InvalidIndices) {
  Test_InvalidIndices_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_InvalidIndices_WithZeroPoints<Int4x2, float, int32_t>();
}

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis0_WithZeroPoints() {
  std::vector<int> data = {-8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5, -8,
                           -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4,
                           0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0,
                           4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4,
                           4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4,
                           -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4};
  std::vector<int64_t> data_shape = {2, 3, 17};
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                               2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f};
  std::vector<int64_t> scales_shape = {2, 3, 2};
  std::vector<int> zero_points = {-1, 1, 0, 0, 1, -1,
                                  1, -1, 1, 0, -1, 1};
  std::vector<float> output = {6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12, 10,
                               3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 4,
                               -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0, -5};
  std::vector<int64_t> output_shape = {1, 3, 17};

  RunGatherBlockQuantized(ToType<T1>(data),
                          data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          ToType<T1>(zero_points),
                          0,
                          2,
                          16,
                          ToType<T2>(output),
                          output_shape,
                          OpTester::ExpectResult::kExpectSuccess);
  RunGatherBlockQuantized(ToType<T1>(data),
                          data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          ToType<T1>(zero_points),
                          -3,
                          -1,
                          16,
                          ToType<T2>(output),
                          output_shape,
                          OpTester::ExpectResult::kExpectSuccess);
}

TEST(GatherBlockQuantizedOpTest, GatherAxis0WithZeroPoints) {
  Test_GatherAxis0_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_GatherAxis0_WithZeroPoints<Int4x2, float, int32_t>();
  Test_GatherAxis0_WithZeroPoints<UInt4x2, MLFloat16, int32_t>();
  Test_GatherAxis0_WithZeroPoints<Int4x2, MLFloat16, int32_t>();
  Test_GatherAxis0_WithZeroPoints<UInt4x2, float, int64_t>();
  Test_GatherAxis0_WithZeroPoints<Int4x2, float, int64_t>();
  Test_GatherAxis0_WithZeroPoints<UInt4x2, MLFloat16, int64_t>();
  Test_GatherAxis0_WithZeroPoints<Int4x2, MLFloat16, int64_t>();
}

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis0_NoZeroPoints() {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};
  std::vector<float> output = {8.f, 10.f, 12.f, 14.f,
                               4.f, 5.f, 6.f, 7.f,
                               -8.f, -6.f, -4.f, -2.f};
  std::vector<int64_t> output_shape = {1, 3, 4};

  RunGatherBlockQuantized(ToType<T1>(data),
                          data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          {},
                          0,
                          2,
                          16,
                          ToType<T2>(output),
                          output_shape,
                          OpTester::ExpectResult::kExpectSuccess);
  RunGatherBlockQuantized(ToType<T1>(data),
                          data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          {},
                          -3,
                          -1,
                          16,
                          ToType<T2>(output),
                          output_shape,
                          OpTester::ExpectResult::kExpectSuccess);
}

TEST(GatherBlockQuantizedOpTest, GatherAxis0NoZeroPoints) {
  Test_GatherAxis0_NoZeroPoints<Int4x2, float, int32_t>();
  Test_GatherAxis0_NoZeroPoints<Int4x2, MLFloat16, int32_t>();
  Test_GatherAxis0_NoZeroPoints<Int4x2, float, int64_t>();
  Test_GatherAxis0_NoZeroPoints<Int4x2, MLFloat16, int64_t>();
}

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis1_WithZeroPoints() {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  std::vector<int> indices = {2, -3, 2};
  std::vector<int64_t> indices_shape = {1, 3};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 1, 4};
  std::vector<int> zero_points = {-1, 1, 0, 0, 1, -1, 0, 0};
  std::vector<float> output = {1.f, 0.f, 2.f, 6.f,
                               -7.f, -16.f, -6.f, -10.f,
                               1.f, 0.f, 2.f, 6.f,
                               -5.f, -4.f, -2.f, -2.f,
                               3.f, 12.f, 6.f, 14.f,
                               -5.f, -4.f, -2.f, -2.f};
  std::vector<int64_t> output_shape = {2, 1, 3, 4};

  RunGatherBlockQuantized(ToType<T1>(data),
                          data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          ToType<T1>(zero_points),
                          1,
                          1,
                          16,
                          ToType<T2>(output),
                          output_shape,
                          OpTester::ExpectResult::kExpectSuccess);
  RunGatherBlockQuantized(ToType<T1>(data),
                          data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          ToType<T1>(zero_points),
                          -2,
                          -2,
                          16,
                          ToType<T2>(output),
                          output_shape,
                          OpTester::ExpectResult::kExpectSuccess);
}

TEST(GatherBlockQuantizedOpTest, GatherAxis1) {
  Test_GatherAxis1_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_GatherAxis1_WithZeroPoints<Int4x2, float, int32_t>();
  Test_GatherAxis1_WithZeroPoints<UInt4x2, MLFloat16, int32_t>();
  Test_GatherAxis1_WithZeroPoints<Int4x2, MLFloat16, int32_t>();
  Test_GatherAxis1_WithZeroPoints<UInt4x2, float, int64_t>();
  Test_GatherAxis1_WithZeroPoints<Int4x2, float, int64_t>();
  Test_GatherAxis1_WithZeroPoints<UInt4x2, MLFloat16, int64_t>();
  Test_GatherAxis1_WithZeroPoints<Int4x2, MLFloat16, int64_t>();
}

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis2_WithZeroPoints() {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  std::vector<int> indices = {-2, 0};
  std::vector<int64_t> indices_shape = {2, 1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f,
                               1.0f, 2.0f, 1.0f, 2.0f,
                               1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {1, 3, 4};
  std::vector<int> zero_points = {-1, 1, 0, 0,
                                  1, -1, 0, 0,
                                  0, 0, 1, -1};
  std::vector<float> output = {-6.f, -7.f, -2.f, -5.f, 1.f, 0.f,
                               6.f, 5.f, 6.f, 3.f, -3.f, -4.f};
  std::vector<int64_t> output_shape = {2, 3, 2, 1};

  RunGatherBlockQuantized(ToType<T1>(data),
                          data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          ToType<T1>(zero_points),
                          2,
                          0,
                          16,
                          ToType<T2>(output),
                          output_shape,
                          OpTester::ExpectResult::kExpectSuccess);
  RunGatherBlockQuantized(ToType<T1>(data),
                          data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          ToType<T1>(zero_points),
                          -1,
                          -3,
                          16,
                          ToType<T2>(output),
                          output_shape,
                          OpTester::ExpectResult::kExpectSuccess);
}

TEST(GatherBlockQuantizedOpTest, GatherAxis2) {
  Test_GatherAxis2_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_GatherAxis2_WithZeroPoints<Int4x2, float, int32_t>();
  Test_GatherAxis2_WithZeroPoints<UInt4x2, MLFloat16, int32_t>();
  Test_GatherAxis2_WithZeroPoints<Int4x2, MLFloat16, int32_t>();
  Test_GatherAxis2_WithZeroPoints<UInt4x2, float, int64_t>();
  Test_GatherAxis2_WithZeroPoints<Int4x2, float, int64_t>();
  Test_GatherAxis2_WithZeroPoints<UInt4x2, MLFloat16, int64_t>();
  Test_GatherAxis2_WithZeroPoints<Int4x2, MLFloat16, int64_t>();
}

}  // namespace test
}  // namespace onnxruntime
