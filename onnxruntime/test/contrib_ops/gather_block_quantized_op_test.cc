// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <vector>
#include <type_traits>
#include <memory>
#include <utility>
#include <sstream>

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// When uint8_t data type is used GatherBlockQuantize applies MatMulNBit's conventions for storing the data.
// That is when no zero points are specified a default zero point of 8 (for 4 bits) or 128 (for 8 bits) is used.
// This convertor hence compensates for that by adding it to the data values, so that the outputs match the results
// that we seen with non uint8_t data types.
// Since both weight and zero_point have same offset, the offset will not impact the value of dequantization:
// `(weight - zero_point) * scale` has same value as `((weight + offset) - (zero_point + offset)) * scale`.
void PackDataForUint8TypeIfNecessary(std::vector<int>& data, std::vector<int64_t>& data_shape, int bits = 4) {
  int64_t total_elements = 1;
  for (const auto& dim : data_shape) {
    total_elements *= dim;
  }
  int64_t input_columns = data_shape.back();
  int64_t total_rows = total_elements / input_columns;

  std::vector<int> packed_data;

  if (bits == 4) {
    // For uint8_t, we need to pack each pair of 4 bits (after adding 8) into a single uint8_t
    int64_t output_columns = (input_columns + 1) / 2;
    packed_data.reserve(total_rows * output_columns);
    for (int64_t row = 0; row < total_rows; ++row) {
      for (int64_t col = 0; col < input_columns; col += 2) {
        int low_nibble = (data[row * input_columns + col] + 8) & 0xF;
        int high_nibble = ((col + 1) < input_columns) ? ((data[row * input_columns + col + 1] + 8) & 0xF) : 0;
        int packed = (high_nibble << 4) | low_nibble;
        packed_data.push_back(packed);
      }
    }
    data_shape.back() = output_columns;
  } else {
    for (auto v : data) {
      packed_data.push_back(v + 128);
    }
  }

  data = packed_data;
}

template <typename T>
std::string VectorToString(const std::vector<T>& vec) {
  std::ostringstream oss;
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i != vec.size() - 1) {
      oss << ",";
    }
  }
  return oss.str();
}

template <typename T>
void CheckDataAndShape(const std::vector<T>& data, const std::vector<int64_t>& shape, std::string name = "") {
  int64_t total_elements = 1;
  for (const auto& dim : shape) {
    total_elements *= dim;
  }

  // UInt4x2 and Int4x2 uses global packing instead of per-row packing.
  if constexpr (std::is_same<T, UInt4x2>::value || std::is_same<T, Int4x2>::value) {
    total_elements = (total_elements + 1) / 2;
  }

  ORT_ENFORCE(static_cast<int64_t>(data.size()) == total_elements, "Data size does not match the shape",
              "Data size: ", data.size(), ", Expected size: ", total_elements,
              ", Shape: ", VectorToString(shape), " Name:", name, " Type:", typeid(T).name());
}

// Combinations: types, gather_axis, quantize_axis, block_size, indices, scale shape vs data shape
template <typename T1, typename T2, typename Tind>
void RunGatherBlockQuantized(const std::vector<T1>& data,
                             const std::vector<int64_t>& data_shape,
                             const std::vector<Tind>& indices,
                             const std::vector<int64_t>& indices_shape,
                             const std::vector<T2>& scales,
                             const std::vector<int64_t>& scales_shape,
                             const std::vector<T1>& zero_points,
                             const std::vector<int64_t>& zero_points_shape,
                             const int64_t gather_axis,
                             const int64_t quantize_axis,
                             const int64_t block_size,
                             const int64_t bits,
                             const std::vector<T2>& output,
                             const std::vector<int64_t>& output_shape,
                             OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                             bool touch_on_device_data = false) {
  CheckDataAndShape<T1>(data, data_shape, "data in RunGatherBlockQuantized");
  CheckDataAndShape<Tind>(indices, indices_shape, "indices in RunGatherBlockQuantized");
  CheckDataAndShape<T2>(scales, scales_shape, "scales in RunGatherBlockQuantized");
  if (!zero_points_shape.empty()) {
    CheckDataAndShape<T1>(zero_points, zero_points_shape, "zero_points in RunGatherBlockQuantized");
  }
  CheckDataAndShape<T2>(output, output_shape, "output in RunGatherBlockQuantized");

  auto run_test = [&](bool indices_is_initializer) {
    OpTester test("GatherBlockQuantized", 1, kMSDomain);

    test.AddAttribute<int64_t>("gather_axis", gather_axis);
    test.AddAttribute<int64_t>("quantize_axis", quantize_axis);
    test.AddAttribute<int64_t>("block_size", block_size);
    test.AddAttribute<int64_t>("bits", bits);

    test.AddInput<T1>("data", data_shape, data);
    test.AddInput<Tind>("indices", indices_shape, indices, indices_is_initializer);
    test.AddInput<T2>("scales", scales_shape, scales);
    if (!zero_points.empty()) {
      test.AddInput<T1>("zero_points", zero_points_shape, zero_points);
    }

    test.AddOutput<T2>("output", output_shape, output);

    if (touch_on_device_data) {
      // test would need to see data on device
      test.Run(expect_result, "", {kWebGpuExecutionProvider}, nullptr);
    } else {
      test.Run(expect_result, "");
    }
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
  using UnpackedType = T::UnpackedType;

  // UInt4x2 and Int4x2 uses global packing instead of per-row packing.
  size_t i = 0;
  constexpr UnpackedType offset = std::is_same<T, Int4x2>::value ? 0 : 8;
  std::vector<T> result;
  for (i = 0; i + 1 < vec.size(); i += 2) {
    result.push_back(T(static_cast<UnpackedType>(vec[i] + offset), static_cast<UnpackedType>(vec[i + 1] + offset)));
  }
  if (i < vec.size()) {
    result.push_back(T(static_cast<UnpackedType>(vec[i] + offset), static_cast<UnpackedType>(0 + offset)));
  }
  return result;
}

// The data and zero_points are not packed
template <typename T1, typename T2, typename Tind>
void RunUnpackedData(
    const std::vector<int>& unpacked_data,
    const std::vector<int64_t>& unpacked_data_shape,
    const std::vector<int>& indices,
    const std::vector<int64_t>& indices_shape,
    const std::vector<float>& scales,
    const std::vector<int64_t>& scales_shape,
    std::vector<int>& zero_points,
    const int64_t gather_axis,
    const int64_t quantize_axis,
    const int64_t block_size,
    const int64_t bits,
    const std::vector<float>& output,
    const std::vector<int64_t>& output_shape,
    bool expect_success,
    bool touch_on_device_data = false) {
  CheckDataAndShape<int>(unpacked_data, unpacked_data_shape, "unpacked_data");
  CheckDataAndShape<int>(indices, indices_shape, "indices");
  CheckDataAndShape<float>(scales, scales_shape, "scales");
  if (!zero_points.empty()) {
    CheckDataAndShape<int>(zero_points, scales_shape, "zero_points");
  }
  CheckDataAndShape<float>(output, output_shape, "output");

  // Make a copy to avoid modifying the original unpacked data.
  std::vector<int> packed_data = unpacked_data;
  std::vector<int64_t> packed_data_shape = unpacked_data_shape;
  if (std::is_same_v<T1, uint8_t>) {
    PackDataForUint8TypeIfNecessary(packed_data, packed_data_shape, static_cast<int>(bits));
  }

  auto expect_result = expect_success ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure;
  if (zero_points.empty()) {
    // If no zero points are provided, we can skip packing them.
    RunGatherBlockQuantized(ToType<T1>(packed_data),
                            packed_data_shape,
                            ToType<Tind>(indices),
                            indices_shape,
                            ToType<T2>(scales),
                            scales_shape,
                            {},
                            {},
                            gather_axis,
                            quantize_axis,
                            block_size,
                            bits,
                            ToType<T2>(output),
                            output_shape,
                            expect_result,
                            touch_on_device_data);
    return;
  }

  // Make a copy to avoid modifying the original unpacked data.
  std::vector<int> packed_zero_point = zero_points;
  std::vector<int64_t> packed_zero_point_shape = scales_shape;
  if (std::is_same_v<T1, uint8_t>) {
    PackDataForUint8TypeIfNecessary(packed_zero_point, packed_zero_point_shape, static_cast<int>(bits));
  }

  RunGatherBlockQuantized(ToType<T1>(packed_data),
                          packed_data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          ToType<T1>(packed_zero_point),
                          packed_zero_point_shape,
                          gather_axis,
                          quantize_axis,
                          block_size,
                          bits,
                          ToType<T2>(output),
                          output_shape,
                          expect_result,
                          touch_on_device_data);
}

template <typename T1, typename T2, typename Tind>
void Test_Fail_WithZeroPoints(int64_t gather_axis,
                              int64_t quantize_axis,
                              int64_t block_size,
                              int64_t bits = 4) {
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

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, false);
}

TEST(GatherBlockQuantizedOpTest, UnsupportedTypes) {
  Test_Fail_WithZeroPoints<int8_t, float, int32_t>(0, 2, 16);
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
  Test_Fail_WithZeroPoints<uint8_t, float, int16_t>(0, 2, 16);
}

template <typename T1, typename T2, typename Tind>
void Test_Fail_WithoutZeroPoints(int64_t gather_axis,
                                 int64_t quantize_axis,
                                 int64_t block_size,
                                 int64_t bits = 4) {
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
  std::vector<int> zero_points = {};
  std::vector<float> output = {8.f, 10.f, 12.f, 14.f,
                               3.f, 4.f, 5.f, 6.f,
                               -6.f, -4.f, -2.f, 0.f};
  std::vector<int64_t> output_shape = {1, 3, 4};

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, false);
}

TEST(GatherBlockQuantizedOpTest, UnsupportedUInt8DataType) {
  // Gather on axis other than 0 is not supported with uint8_t
  Test_Fail_WithoutZeroPoints<uint8_t, float, int32_t>(1, 2, 16);
  Test_Fail_WithoutZeroPoints<uint8_t, float, int16_t>(1, 2, 16);
}

TEST(GatherBlockQuantizedOpTest, InvalidBlockSize) {
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 8);
  Test_Fail_WithZeroPoints<Int4x2, float, int32_t>(0, 2, 17);
  Test_Fail_WithZeroPoints<uint8_t, float, int32_t>(0, 2, 17);
}

TEST(GatherBlockQuantizedOpTest, InvalidGatherAxis) {
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(3, 2, 16);
  Test_Fail_WithZeroPoints<Int4x2, float, int32_t>(-4, 2, 16);
  Test_Fail_WithZeroPoints<uint8_t, float, int32_t>(-4, 2, 16);
}

TEST(GatherBlockQuantizedOpTest, InvalidQuantizeAxis) {
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 3, 16);
  Test_Fail_WithZeroPoints<Int4x2, float, int32_t>(0, -4, 16);
  Test_Fail_WithZeroPoints<uint8_t, float, int32_t>(0, -4, 16);
}

TEST(GatherBlockQuantizedOpTest, NotSupportedBits) {
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 1);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 2);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 3);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 5);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 6);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 7);
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

  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 2;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 4;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, false);
}

TEST(GatherBlockQuantizedOpTest, ShapeMismatch) {
  Test_ShapeMismatch_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_ShapeMismatch_WithZeroPoints<Int4x2, float, int32_t>();
  Test_ShapeMismatch_WithZeroPoints<uint8_t, float, int32_t>();
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

  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 2;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 4;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, false, true);
}

TEST(GatherBlockQuantizedOpTest, InvalidIndices) {
  Test_InvalidIndices_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_InvalidIndices_WithZeroPoints<Int4x2, float, int32_t>();
  Test_InvalidIndices_WithZeroPoints<uint8_t, float, int32_t>();
}

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis0_WithZeroPoints(int bits = 4) {
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

  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 2;
  constexpr int64_t block_size = 16;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                -3, -1, block_size, bits, output, output_shape, true);
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
void Test_GatherAxis0_WithZeroPoints_Uint8(int bits = 4) {
  std::vector<int> data = {-8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5, -8, 0,
                           -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4, 0,
                           0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0,
                           4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 0,
                           4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 0,
                           -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4, 0};
  std::vector<int64_t> data_shape = {2, 3, 18};
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                               2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f};
  std::vector<int64_t> scales_shape = {2, 3, 2};
  std::vector<int> zero_points = {-1, 1, 0, 0, 1, -1,
                                  1, -1, 1, 0, -1, 1};
  // 4 bits output
  std::vector<float> output = {6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12, 10, 2,
                               3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 4, 0,
                               -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0, -5, -1};
  std::vector<int64_t> output_shape = {1, 3, 18};

  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 2;
  constexpr int64_t block_size = 16;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                -3, -1, block_size, bits, output, output_shape, true);
}

TEST(GatherBlockQuantizedOpTest, GatherAxis0WithZeroPoints_4Bits) {
  Test_GatherAxis0_WithZeroPoints_Uint8<uint8_t, float, int32_t>();
  Test_GatherAxis0_WithZeroPoints_Uint8<uint8_t, MLFloat16, int64_t>();
}

TEST(GatherBlockQuantizedOpTest, GatherAxis0WithZeroPoints_8Bits) {
  Test_GatherAxis0_WithZeroPoints_Uint8<uint8_t, float, int32_t>(8);
  Test_GatherAxis0_WithZeroPoints_Uint8<uint8_t, MLFloat16, int64_t>(8);
}

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis0_NoZeroPoints(int bits = 4) {
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

  // 4 bits output
  std::vector<float> output = {8.f, 10.f, 12.f, 14.f,
                               4.f, 5.f, 6.f, 7.f,
                               -8.f, -6.f, -4.f, -2.f};

  std::vector<int64_t> output_shape = {1, 3, 4};

  std::vector<int> zero_points = {};
  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 2;
  constexpr int64_t block_size = 16;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                -3, -1, block_size, bits, output, output_shape, true);
}

TEST(GatherBlockQuantizedOpTest, GatherAxis0NoZeroPoints) {
  Test_GatherAxis0_NoZeroPoints<Int4x2, float, int32_t>();
  Test_GatherAxis0_NoZeroPoints<Int4x2, MLFloat16, int32_t>();
  Test_GatherAxis0_NoZeroPoints<Int4x2, float, int64_t>();
  Test_GatherAxis0_NoZeroPoints<Int4x2, MLFloat16, int64_t>();
}

TEST(GatherBlockQuantizedOpTest, GatherAxis0NoZeroPoints_4Bits) {
  Test_GatherAxis0_NoZeroPoints<uint8_t, float, int32_t>();
  Test_GatherAxis0_NoZeroPoints<uint8_t, MLFloat16, int32_t>();
  Test_GatherAxis0_NoZeroPoints<uint8_t, float, int64_t>();
  Test_GatherAxis0_NoZeroPoints<uint8_t, MLFloat16, int64_t>();
}

TEST(GatherBlockQuantizedOpTest, GatherAxis0NoZeroPoints_8Bits) {
  Test_GatherAxis0_NoZeroPoints<uint8_t, float, int64_t>(8);
  Test_GatherAxis0_NoZeroPoints<uint8_t, MLFloat16, int64_t>(8);
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

  constexpr int64_t gather_axis = 1;
  constexpr int64_t quantize_axis = 1;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 4;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                -2, -2, block_size, bits, output, output_shape, true);
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

  constexpr int64_t gather_axis = 2;
  constexpr int64_t quantize_axis = 0;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 4;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                -1, -3, block_size, bits, output, output_shape, true);
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
