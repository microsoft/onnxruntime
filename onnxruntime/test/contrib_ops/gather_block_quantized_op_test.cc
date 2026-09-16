// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <vector>
#include <type_traits>
#include <memory>
#include <utility>
#include <sstream>
#include <unordered_set>
#include <string>

#ifndef _WIN32
#include <sys/mman.h>
#endif

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "test/common/cuda_op_test_utils.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_CUDA
#include "contrib_ops/cuda/quantization/gather_block_quantized.h"
#include "core/graph/model.h"
#include "core/platform/env.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/temp_dir.h"
#ifndef BUILD_CUDA_EP_AS_PLUGIN
#include "core/providers/cuda/cuda_execution_provider_info.h"
#endif
#endif

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
  } else if (bits == 2) {
    // For uint8_t with bits=2, pack 4 elements (2 bits each, after adding 2) into a single uint8_t.
    // Element index 0 occupies the lowest 2 bits; index 3 occupies the highest 2 bits.
    int64_t output_columns = (input_columns + 3) / 4;
    packed_data.reserve(total_rows * output_columns);
    for (int64_t row = 0; row < total_rows; ++row) {
      for (int64_t col = 0; col < input_columns; col += 4) {
        int packed = 0;
        for (int k = 0; k < 4; ++k) {
          if ((col + k) < input_columns) {
            int v = (data[row * input_columns + col + k] + 2) & 0x3;
            packed |= (v << (2 * k));
          }
        }
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
              ", Shape: ", VectorToString(shape), " Name:", name);
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
  (void)touch_on_device_data;
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

    bool enable_cuda = HasCudaEnvironment(0);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    if (enable_cuda) {
      eps.push_back(DefaultCudaExecutionProvider());
    } else {
      eps.push_back(DefaultCpuExecutionProvider());
    }

    test.Run(expect_result, "", {}, nullptr, &eps);
  };

  run_test(false);
  run_test(true);
}

// WebGPU-specific runner for GatherBlockQuantized. Only supports uint8 data with gather_axis == 0.
template <typename T2, typename Tind>
void RunGatherBlockQuantizedWebGpu(const std::vector<uint8_t>& data,
                                   const std::vector<int64_t>& data_shape,
                                   const std::vector<Tind>& indices,
                                   const std::vector<int64_t>& indices_shape,
                                   const std::vector<T2>& scales,
                                   const std::vector<int64_t>& scales_shape,
                                   const std::vector<uint8_t>& zero_points,
                                   const std::vector<int64_t>& zero_points_shape,
                                   const int64_t gather_axis,
                                   const int64_t quantize_axis,
                                   const int64_t block_size,
                                   const int64_t bits,
                                   const std::vector<T2>& output,
                                   const std::vector<int64_t>& output_shape,
                                   OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess) {
#ifdef USE_WEBGPU
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    return;
  }

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", gather_axis);
  test.AddAttribute<int64_t>("quantize_axis", quantize_axis);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", bits);

  test.AddInput<uint8_t>("data", data_shape, data);
  test.AddInput<Tind>("indices", indices_shape, indices);
  test.AddInput<T2>("scales", scales_shape, scales);
  if (!zero_points.empty()) {
    test.AddInput<uint8_t>("zero_points", zero_points_shape, zero_points);
  }
  test.AddOutput<T2>("output", output_shape, output);

  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultWebGpuExecutionProvider());
  test.Run(expect_result, "", {}, nullptr, &eps);
#else
  (void)data;
  (void)data_shape;
  (void)indices;
  (void)indices_shape;
  (void)scales;
  (void)scales_shape;
  (void)zero_points;
  (void)zero_points_shape;
  (void)gather_axis;
  (void)quantize_axis;
  (void)block_size;
  (void)bits;
  (void)output;
  (void)output_shape;
  (void)expect_result;
#endif
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
  using UnpackedType = typename T::UnpackedType;

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

#ifndef USE_CUDA
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
#endif

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

#ifndef USE_CUDA
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
  Test_Fail_WithZeroPoints<uint8_t, float, int32_t>(0, 2, 16, 0);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 0);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 1);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 2);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 3);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 5);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 6);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 7);
}
#endif

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

#ifndef USE_CUDA
TEST(GatherBlockQuantizedOpTest, ShapeMismatch) {
  Test_ShapeMismatch_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_ShapeMismatch_WithZeroPoints<Int4x2, float, int32_t>();
  Test_ShapeMismatch_WithZeroPoints<uint8_t, float, int32_t>();
}
#endif

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
  output.assign(output.size(), 0.0f);
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true, true);
}

template <typename T1, typename T2, typename Tind>
void Test_NegativeInvalidIndices_WithZeroPoints() {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  std::vector<int> indices = {-3};
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
  output.assign(output.size(), 0.0f);
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true, true);
}

TEST(GatherBlockQuantizedOpTest, InvalidIndices) {
  Test_InvalidIndices_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_InvalidIndices_WithZeroPoints<Int4x2, float, int32_t>();
  Test_InvalidIndices_WithZeroPoints<uint8_t, float, int32_t>();
}

#ifdef USE_CUDA
TEST(GatherBlockQuantizedOpTest, InvalidIndicesZeroFillCuda) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "CUDA not available";
  }

  Test_InvalidIndices_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_InvalidIndices_WithZeroPoints<UInt4x2, float, int64_t>();
  Test_InvalidIndices_WithZeroPoints<uint8_t, float, int32_t>();
}

TEST(GatherBlockQuantizedOpTest, NegativeInvalidIndicesZeroFillCuda) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "CUDA not available";
  }

  Test_NegativeInvalidIndices_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_NegativeInvalidIndices_WithZeroPoints<UInt4x2, float, int64_t>();
  Test_NegativeInvalidIndices_WithZeroPoints<uint8_t, float, int32_t>();
}
#endif

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

#ifndef USE_CUDA
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
#endif

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

#ifndef USE_CUDA
TEST(GatherBlockQuantizedOpTest, GatherAxis0WithZeroPoints_4Bits) {
  Test_GatherAxis0_WithZeroPoints_Uint8<uint8_t, float, int32_t>();
  Test_GatherAxis0_WithZeroPoints_Uint8<uint8_t, MLFloat16, int64_t>();
}

TEST(GatherBlockQuantizedOpTest, GatherAxis0WithZeroPoints_8Bits) {
  Test_GatherAxis0_WithZeroPoints_Uint8<uint8_t, float, int32_t>(8);
  Test_GatherAxis0_WithZeroPoints_Uint8<uint8_t, MLFloat16, int64_t>(8);
}
#endif

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

#ifndef USE_CUDA
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
#endif

#ifdef USE_CUDA
TEST(GatherBlockQuantizedOpTest, GatherAxis0NoZeroPoints_4Bits_Cuda) {
  const std::vector<uint8_t> data(32, 0xAA);
  const std::vector<float> scales = {0.5f, 0.25f};
  std::vector<float> output(32, 1.0f);
  output.insert(output.end(), 32, 0.5f);

  RunGatherBlockQuantized<uint8_t, float, int64_t>(
      data, {2, 16}, {0, 1}, {2}, scales, {2, 1}, {}, {},
      0, 1, 32, 4, output, {2, 32});
}
#endif

#ifndef USE_CUDA
TEST(GatherBlockQuantizedOpTest, GatherAxis0NoZeroPoints_2Bits_Uint8) {
  // 2-bit signed values in {-2, -1, 0, 1}. The test infra adds an offset of 2 when packing
  // and the kernel uses default zero_point = 2^(bits-1) = 2, so the dequantized value matches.
  // Block size 16 covers the entire last dim with one scale per row.
  std::vector<int> data = {-2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1,
                           1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2,
                           0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1,
                           -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2};
  std::vector<int64_t> data_shape = {2, 3, 16};
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};

  // indices = [1] -> pick outer index 1, so we expect rows 3, 4, 5 of the unpacked data above
  // (the second {-1,-2,1,0,...}, {1,1,...}, {-2,-2,...} block), each scaled by
  // scales[3], scales[4], scales[5] = 2.0, 1.0, 2.0.
  std::vector<float> output = {-2.f, -4.f, 2.f, 0.f, -2.f, -4.f, 2.f, 0.f, -2.f, -4.f, 2.f, 0.f, -2.f, -4.f, 2.f, 0.f,
                               1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                               -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f};
  std::vector<int64_t> output_shape = {1, 3, 16};

  std::vector<int> zero_points = {};
  RunUnpackedData<uint8_t, float, int32_t>(data, data_shape, indices, indices_shape, scales, scales_shape,
                                           zero_points, /*gather_axis=*/0, /*quantize_axis=*/2,
                                           /*block_size=*/16, /*bits=*/2, output, output_shape, true);
}

TEST(GatherBlockQuantizedOpTest, GatherAxis0WithZeroPoints_2Bits_Uint8_PackedZpNotMultipleOf4) {
  // Exercises the 2-bit zero-point row-boundary logic: scale_qaxis_dim = 5 is NOT a multiple
  // of the 2-bit packing factor (4), so each zero_points row occupies (5+3)/4 = 2 bytes and
  // the within-row qaxis index spans both bytes (lanes 0..3 in byte 0, lane 0 in byte 1).
  // The kernel must address the packed byte using (scale_row, q_in_row), not a flat scales
  // offset which would cross row boundaries.
  //
  // Logical layout: data {2, 80}, quantize_axis = 1 (last), block_size = 16, bits = 2.
  // scales {2, 5}; zero_points logical {2, 5} -> packs to {2, 2}.
  std::vector<int> data(2 * 80, 0);  // all zeros; dequant simplifies to (0 - zp) * scale
  std::vector<int64_t> data_shape = {2, 80};
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  // All scales = 1 so dequant value equals -zp directly.
  std::vector<float> scales(2 * 5, 1.0f);
  std::vector<int64_t> scales_shape = {2, 5};
  // Logical 2-bit zero points in {-2,-1,0,1}; helper packs along last dim (5 -> 2 bytes).
  // Row 0: [-2, 1, 0, -1, 1] ; Row 1: [1, 0, -1, -2, 0]
  std::vector<int> zero_points = {-2, 1, 0, -1, 1, 1, 0, -1, -2, 0};

  // indices = [1] -> pick row 1: zp = [1, 0, -1, -2, 0]
  // dequant per block = (0 - zp) * 1 = [-1, 0, 1, 2, 0]
  std::vector<float> output;
  output.reserve(80);
  for (float v : {-1.f, 0.f, 1.f, 2.f, 0.f}) {
    output.insert(output.end(), 16, v);
  }
  std::vector<int64_t> output_shape = {1, 80};

  RunUnpackedData<uint8_t, float, int32_t>(data, data_shape, indices, indices_shape, scales, scales_shape,
                                           zero_points, /*gather_axis=*/0, /*quantize_axis=*/1,
                                           /*block_size=*/16, /*bits=*/2, output, output_shape, true);
}
#endif

#ifdef USE_WEBGPU
TEST(GatherBlockQuantizedOpTest, WebGpu_GatherAxis0NoZeroPoints_2Bits_Uint8) {
  // Same logical data and expectation as the CPU GatherAxis0NoZeroPoints_2Bits_Uint8 test.
  // Logical 2-bit values in {-2, -1, 0, 1}, encoded as v+2 in {0..3} and packed 4 per byte
  // (low-order bits first). 16 logical 2-bit values -> 4 bytes per row.
  // Pack helper:
  auto pack4 = [](int v0, int v1, int v2, int v3) -> uint8_t {
    auto enc = [](int v) { return static_cast<uint8_t>((v + 2) & 0x3); };
    return static_cast<uint8_t>(enc(v0) | (enc(v1) << 2) | (enc(v2) << 4) | (enc(v3) << 6));
  };

  // Build packed data: shape {2, 3, 4} (16 logical elements per row -> 4 bytes).
  std::vector<uint8_t> data;
  data.reserve(2 * 3 * 4);
  auto push_row = [&](std::vector<int> row) {
    ORT_ENFORCE(row.size() == 16);
    for (size_t i = 0; i < 16; i += 4) {
      data.push_back(pack4(row[i], row[i + 1], row[i + 2], row[i + 3]));
    }
  };
  push_row({-2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1});
  push_row({1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2});
  push_row({0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1});
  push_row({-1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0});
  push_row({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  push_row({-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2});

  std::vector<int64_t> data_shape = {2, 3, 4};
  std::vector<int32_t> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};

  std::vector<float> output = {-2.f, -4.f, 2.f, 0.f, -2.f, -4.f, 2.f, 0.f, -2.f, -4.f, 2.f, 0.f, -2.f, -4.f, 2.f, 0.f,
                               1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                               -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f};
  std::vector<int64_t> output_shape = {1, 3, 16};

  std::vector<uint8_t> zero_points = {};
  std::vector<int64_t> zero_points_shape = {};
  RunGatherBlockQuantizedWebGpu<float, int32_t>(data, data_shape, indices, indices_shape, scales, scales_shape,
                                                zero_points, zero_points_shape,
                                                /*gather_axis=*/0, /*quantize_axis=*/2,
                                                /*block_size=*/16, /*bits=*/2, output, output_shape);
}

TEST(GatherBlockQuantizedOpTest, WebGpu_EmptyIndices_8Bits_Uint8) {
  // An empty indices tensor produces an empty output. The kernel must not dispatch a
  // (0, 1, 1) workgroup in that case. See issue #28772.
  std::vector<uint8_t> data(2 * 16, 128);
  std::vector<int64_t> data_shape = {2, 16};
  std::vector<int32_t> indices = {};
  std::vector<int64_t> indices_shape = {0};
  std::vector<float> scales = {1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 1};

  std::vector<float> output = {};
  std::vector<int64_t> output_shape = {0, 16};

  std::vector<uint8_t> zero_points = {};
  std::vector<int64_t> zero_points_shape = {};
  RunGatherBlockQuantizedWebGpu<float, int32_t>(data, data_shape, indices, indices_shape, scales, scales_shape,
                                                zero_points, zero_points_shape,
                                                /*gather_axis=*/0, /*quantize_axis=*/1,
                                                /*block_size=*/16, /*bits=*/8, output, output_shape);
}

TEST(GatherBlockQuantizedOpTest, WebGpu_InvalidIndices_2Bits_Uint8) {
  auto pack4 = [](int v0, int v1, int v2, int v3) -> uint8_t {
    auto enc = [](int v) { return static_cast<uint8_t>((v + 2) & 0x3); };
    return static_cast<uint8_t>(enc(v0) | (enc(v1) << 2) | (enc(v2) << 4) | (enc(v3) << 6));
  };

  std::vector<uint8_t> data;
  data.reserve(2 * 3 * 4);
  auto push_row = [&](std::vector<int> row) {
    ORT_ENFORCE(row.size() == 16);
    for (size_t i = 0; i < 16; i += 4) {
      data.push_back(pack4(row[i], row[i + 1], row[i + 2], row[i + 3]));
    }
  };
  push_row({-2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1});
  push_row({1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2});
  push_row({0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1});
  push_row({-1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0, -1, -2, 1, 0});
  push_row({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  push_row({-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2});

  std::vector<int64_t> data_shape = {2, 3, 4};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};
  std::vector<uint8_t> zero_points = {};
  std::vector<int64_t> zero_points_shape = {};
  std::vector<float> output(2 * 3 * 16, 0.0f);
  std::vector<int64_t> output_shape = {2, 3, 16};

  std::vector<int32_t> indices_i32 = {2, -3};
  std::vector<int64_t> indices_shape = {2};
  RunGatherBlockQuantizedWebGpu<float, int32_t>(data, data_shape, indices_i32, indices_shape, scales, scales_shape,
                                                zero_points, zero_points_shape,
                                                /*gather_axis=*/0, /*quantize_axis=*/2,
                                                /*block_size=*/16, /*bits=*/2, output, output_shape);

  std::vector<int64_t> indices_i64 = {2, -3};
  RunGatherBlockQuantizedWebGpu<float, int64_t>(data, data_shape, indices_i64, indices_shape, scales, scales_shape,
                                                zero_points, zero_points_shape,
                                                /*gather_axis=*/0, /*quantize_axis=*/2,
                                                /*block_size=*/16, /*bits=*/2, output, output_shape);
}

TEST(GatherBlockQuantizedOpTest, WebGpu_GatherAxis0WithZeroPoints_2Bits_Uint8_PackedZpNotMultipleOf4) {
  // WebGPU companion to GatherAxis0WithZeroPoints_2Bits_Uint8_PackedZpNotMultipleOf4.
  // scale_qaxis_dim = 5 (not a multiple of 4); packed zero_points last dim = (5+3)/4 = 2 bytes.
  // The within-row qaxis index spans both bytes, validating the packed-byte addressing path
  // (scale_row * zp_packed_qaxis_dim + q_idx/4, shift (q_idx%4)*2) in the WebGPU shader.
  auto pack4 = [](int v0, int v1, int v2, int v3) -> uint8_t {
    auto enc = [](int v) { return static_cast<uint8_t>((v + 2) & 0x3); };
    return static_cast<uint8_t>(enc(v0) | (enc(v1) << 2) | (enc(v2) << 4) | (enc(v3) << 6));
  };

  // Packed data: each logical row = 80 zeros -> 20 bytes of pack4(0,0,0,0) = 0xAA. data_shape {2, 20}.
  std::vector<uint8_t> data(2 * 20, pack4(0, 0, 0, 0));
  std::vector<int64_t> data_shape = {2, 20};

  std::vector<int32_t> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales(2 * 5, 1.0f);
  std::vector<int64_t> scales_shape = {2, 5};

  // Packed zero_points: shape {2, 2} (5 logical -> 2 bytes per row).
  //   Row 0 logical [-2, 1, 0, -1, 1] -> byte0 = pack4(-2, 1, 0, -1) = 0x6C; byte1 lane0=1, rest dont-care.
  //   Row 1 logical [ 1, 0,-1,-2, 0] -> byte0 = pack4( 1, 0,-1,-2) = 0x1B; byte1 lane0=0, rest dont-care.
  std::vector<uint8_t> zero_points = {
      pack4(-2, 1, 0, -1), pack4(1, -2, -2, -2),
      pack4(1, 0, -1, -2), pack4(0, -2, -2, -2)};
  std::vector<int64_t> zero_points_shape = {2, 2};

  // Picking row 1 via indices=[1]: dequant per block = (0 - zp) * 1 = [-1, 0, 1, 2, 0].
  std::vector<float> output;
  output.reserve(80);
  for (float v : {-1.f, 0.f, 1.f, 2.f, 0.f}) {
    output.insert(output.end(), 16, v);
  }
  std::vector<int64_t> output_shape = {1, 80};

  RunGatherBlockQuantizedWebGpu<float, int32_t>(data, data_shape, indices, indices_shape, scales, scales_shape,
                                                zero_points, zero_points_shape,
                                                /*gather_axis=*/0, /*quantize_axis=*/1,
                                                /*block_size=*/16, /*bits=*/2, output, output_shape);
}
#endif  // USE_WEBGPU

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis0_QuantizedAxis1_WithZeroPoints_4Bits() {
  // This test case specific to shared 4bit token_embedding/lm_head use case on CUDA
  std::vector<int> data = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
                           0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 16};
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {2.0f, 1.0f};
  std::vector<int64_t> scales_shape = {2, 1};
  // Explicit zero points for each row
  std::vector<int> zero_points = {-2, 1};

  // With explicit zero points:
  // Unpacked data (row 1): [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1] ---add offset 8--->
  // Packed (add offset 8): [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]
  // Gathered scales (row 1): scale = 1.0f, zero_point (row 1): packed: [1] ---add offset 8---> unpacked: [9]
  // Expected (CUDA doesn't subtract zero point): [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]
  std::vector<float> output = {8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  std::vector<int64_t> output_shape = {1, 16};

  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 1;  // Last axis (required for CUDA)
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 4;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);
}

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis0_QuantizedAxis1_WithZeroPoints_8Bits() {
  // This test case specific to shared 8bit token_embedding/lm_head use case on CUDA
  std::vector<int> data = {-128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, -116, -115, -114, -113,
                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int64_t> data_shape = {2, 16};
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 1};
  // Explicit zero points
  std::vector<int> zero_points = {10, -5};

  // With explicit zero points:
  // Unpacked data (row 1): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] ---add offset 128--->
  // Packed (row1): [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143]
  // Zero point unpacked: [-5] ---add offset 128---> packed: [123]
  // Dequantization: [(128-123)*2, (129-123)*2, ..., (143-123)*2] = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
  std::vector<float> output = {10.f, 12.f, 14.f, 16.f, 18.f, 20.f, 22.f, 24.f, 26.f, 28.f, 30.f, 32.f, 34.f, 36.f, 38.f, 40.f};
  std::vector<int64_t> output_shape = {1, 16};

  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 1;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 8;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);
}

#ifdef USE_CUDA
TEST(GatherBlockQuantizedOpTest, GatherAxis0_QuantizedAxis1_Uint8_4Bits_WithZeroPoints) {
  Test_GatherAxis0_QuantizedAxis1_WithZeroPoints_4Bits<uint8_t, float, int32_t>();
  Test_GatherAxis0_QuantizedAxis1_WithZeroPoints_4Bits<uint8_t, MLFloat16, int32_t>();
  Test_GatherAxis0_QuantizedAxis1_WithZeroPoints_4Bits<uint8_t, float, int64_t>();
  Test_GatherAxis0_QuantizedAxis1_WithZeroPoints_4Bits<uint8_t, MLFloat16, int64_t>();
}

TEST(GatherBlockQuantizedOpTest, GatherAxis0_QuantizedAxis1_Uint8_8Bits_WithZeroPoints) {
  Test_GatherAxis0_QuantizedAxis1_WithZeroPoints_8Bits<uint8_t, float, int32_t>();
  Test_GatherAxis0_QuantizedAxis1_WithZeroPoints_8Bits<uint8_t, MLFloat16, int32_t>();
  Test_GatherAxis0_QuantizedAxis1_WithZeroPoints_8Bits<uint8_t, float, int64_t>();
  Test_GatherAxis0_QuantizedAxis1_WithZeroPoints_8Bits<uint8_t, MLFloat16, int64_t>();
}
#endif

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

#ifndef USE_CUDA
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
#endif

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

#ifndef USE_CUDA
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
#endif

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis_WithZeroPoints_NoPading() {
  std::vector<int> data = {
      -8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5,
      -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1,
      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
      4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
      4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7,
      -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1};

  std::vector<int64_t> data_shape = {2, 3, 16};
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};
  std::vector<int> zero_points = {-1, 1, 0, 0, 1, -1};
  std::vector<float> output = {
      8, 10, 12, 14, 8, 10, 12, 14, 8, 10, 12, 14, 8, 10, 12, 14,
      3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6,
      -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0};
  std::vector<int64_t> output_shape = {1, 3, 16};

  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 2;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 4;

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);
}

#ifdef USE_CUDA
TEST(GatherBlockQuantizedOpTest, GatherAxisWithZeroPointsNoPading) {
  Test_GatherAxis_WithZeroPoints_NoPading<Int4x2, float, int32_t>();
  Test_GatherAxis_WithZeroPoints_NoPading<Int4x2, MLFloat16, int32_t>();
  Test_GatherAxis_WithZeroPoints_NoPading<Int4x2, float, int64_t>();
  Test_GatherAxis_WithZeroPoints_NoPading<Int4x2, MLFloat16, int64_t>();
  Test_GatherAxis_WithZeroPoints_NoPading<UInt4x2, float, int32_t>();
  Test_GatherAxis_WithZeroPoints_NoPading<UInt4x2, MLFloat16, int32_t>();
  Test_GatherAxis_WithZeroPoints_NoPading<UInt4x2, float, int64_t>();
  Test_GatherAxis_WithZeroPoints_NoPading<UInt4x2, MLFloat16, int64_t>();
}
#endif

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis_NoPading_4bit() {
  std::vector<int> data = {
      -8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5,
      -8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5,
      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
      7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4,
      7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4};

  std::vector<int64_t> data_shape = {2, 3, 16};
  std::vector<int> indices = {0};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};
  std::vector<int> zero_points = {};
  std::vector<float> output = {
      -8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5,
      -16, -14, -12, -10, -16, -14, -12, -10, -16, -14, -12, -10, -16, -14, -12, -10,
      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  std::vector<int64_t> output_shape = {1, 3, 16};

  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 2;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 4;

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);
}

#ifdef USE_CUDA
TEST(GatherBlockQuantizedOpTest, GatherAxisNoPadingUInt8_4Bits) {
  Test_GatherAxis_NoPading_4bit<uint8_t, float, int32_t>();
  Test_GatherAxis_NoPading_4bit<uint8_t, MLFloat16, int32_t>();
  Test_GatherAxis_NoPading_4bit<uint8_t, float, int64_t>();
  Test_GatherAxis_NoPading_4bit<uint8_t, MLFloat16, int64_t>();
}
#endif

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis_NoPading_8bit() {
  std::vector<int> data = {
      127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112,
      127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112,
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
      127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112,
      127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112};

  std::vector<int64_t> data_shape = {2, 3, 16};
  std::vector<int> indices = {0};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};
  std::vector<int> zero_points = {};
  std::vector<float> output = {
      127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112,
      254, 252, 250, 248, 246, 244, 242, 240, 238, 236, 234, 232, 230, 228, 226, 224,
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<int64_t> output_shape = {1, 3, 16};

  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 2;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 8;

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);
}

#ifdef USE_CUDA
TEST(GatherBlockQuantizedOpTest, GatherAxisNoPadingUInt8) {
  Test_GatherAxis_NoPading_8bit<uint8_t, float, int32_t>();
  Test_GatherAxis_NoPading_8bit<uint8_t, MLFloat16, int32_t>();
  Test_GatherAxis_NoPading_8bit<uint8_t, float, int64_t>();
  Test_GatherAxis_NoPading_8bit<uint8_t, MLFloat16, int64_t>();
}

TEST(GatherBlockQuantizedOpTest, CudaIntegerDefaults) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<UInt4x2> data(64, UInt4x2(1, 1));
  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("bits", 4);
  test.AddInput<UInt4x2>("data", {1, 128}, data);
  test.AddInput<int64_t>("indices", {1}, {0});
  test.AddInput<float>("scales", {1, 1}, {1.0f});
  test.AddOutput<float>("output", {1, 128}, std::vector<float>(128, 1.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}
#endif

// GatherBlockQuantized also supports gathering rows from an FP8 or FP4 block-scaled constant table
// (no zero point, since FP8/FP4 quantization is symmetric) and dequantizing them:
// output[...] = float(data[...]) * scales[block(...)].
// TensorRT/OpenVINO don't register FP8/FP4 kernels for this op, so those EPs fall back to CPU.
static const std::unordered_set<std::string> kFpExcludedProviders = {
    kTensorrtExecutionProvider, kOpenVINOExecutionProvider};

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(GatherBlockQuantizedOpTest, FpBasicPerRowScale) {
  // data: [4, 4] FP8 E4M3FN. block_size = 0 -> one scale per row (quantize_axis = 1, the whole row).
  std::vector<Float8E4M3FN> data = {
      Float8E4M3FN(1.0f), Float8E4M3FN(2.0f), Float8E4M3FN(4.0f), Float8E4M3FN(8.0f),
      Float8E4M3FN(-1.0f), Float8E4M3FN(-2.0f), Float8E4M3FN(-4.0f), Float8E4M3FN(-8.0f),
      Float8E4M3FN(1.0f), Float8E4M3FN(1.0f), Float8E4M3FN(1.0f), Float8E4M3FN(1.0f),
      Float8E4M3FN(2.0f), Float8E4M3FN(2.0f), Float8E4M3FN(2.0f), Float8E4M3FN(2.0f)};
  std::vector<float> scales = {1.0f, 0.5f, 2.0f, 3.0f};  // shape [4, 1]
  std::vector<int64_t> indices = {1, 3};
  std::vector<float> expected = {
      -0.5f, -1.0f, -2.0f, -4.0f,
      6.0f, 6.0f, 6.0f, 6.0f};

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {4, 4}, data, true);
  test.AddInput<int64_t>("indices", {2}, indices);
  test.AddInput<float>("scales", {4, 1}, scales);
  test.AddOutput<float>("output", {2, 4}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kFpExcludedProviders);
}

TEST(GatherBlockQuantizedOpTest, FpGlobalPerTensorScale) {
  // data: [4, 4] FP8 E4M3FN. scales has shape [1, 1]: a single global scale for the whole table,
  // broadcast along gather_axis (0) and matching the single quantize-axis block. This mirrors an FP8-quantized
  // embedding table that uses one scalar `weight_scale` shared by every row (e.g. HF's
  // FP8Embedding: `rows.to(weight_scale.dtype) * weight_scale`, where `weight_scale` has shape (1,)).
  std::vector<Float8E4M3FN> data = {
      Float8E4M3FN(1.0f), Float8E4M3FN(2.0f), Float8E4M3FN(4.0f), Float8E4M3FN(8.0f),
      Float8E4M3FN(-1.0f), Float8E4M3FN(-2.0f), Float8E4M3FN(-4.0f), Float8E4M3FN(-8.0f),
      Float8E4M3FN(1.0f), Float8E4M3FN(1.0f), Float8E4M3FN(1.0f), Float8E4M3FN(1.0f),
      Float8E4M3FN(2.0f), Float8E4M3FN(2.0f), Float8E4M3FN(2.0f), Float8E4M3FN(2.0f)};
  std::vector<float> scales = {0.5f};  // shape [1, 1], one value for the entire tensor
  std::vector<int64_t> indices = {1, 3};
  std::vector<float> expected = {
      -0.5f, -1.0f, -2.0f, -4.0f,
      1.0f, 1.0f, 1.0f, 1.0f};

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {4, 4}, data, true);
  test.AddInput<int64_t>("indices", {2}, indices);
  test.AddInput<float>("scales", {1, 1}, scales);
  test.AddOutput<float>("output", {2, 4}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kFpExcludedProviders);
}

TEST(GatherBlockQuantizedOpTest, FpEmptyTrailingDimension) {
  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 1);
  test.AddAttribute<int64_t>("quantize_axis", 2);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {2, 3, 0}, {}, true);
  test.AddInput<int64_t>("indices", {1}, {0});
  test.AddInput<float>("scales", {2, 3, 0}, {});
  test.AddOutput<float>("output", {2, 1, 0}, {});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kFpExcludedProviders);
}

TEST(GatherBlockQuantizedOpTest, FpSubRowBlockScale) {
  // data: [1, 32] FP8 E4M3FN, block_size = 16 -> 2 blocks of 16 elements each along quantize_axis = 1.
  // (block_size must be 0 or a power of 2 >= 16, per the operator contract.)
  std::vector<Float8E4M3FN> data(32);
  for (int i = 0; i < 16; ++i) {
    data[static_cast<size_t>(i)] = Float8E4M3FN(1.0f);
  }
  for (int i = 16; i < 32; ++i) {
    data[static_cast<size_t>(i)] = Float8E4M3FN(4.0f);
  }
  std::vector<float> scales = {1.0f, 0.5f};  // shape [1, 2]: one scale per 16-element block
  std::vector<int64_t> indices = {0};
  std::vector<float> expected(32);
  for (int i = 0; i < 16; ++i) {
    expected[static_cast<size_t>(i)] = 1.0f;  // block 0: 1.0 * 1.0
  }
  for (int i = 16; i < 32; ++i) {
    expected[static_cast<size_t>(i)] = 2.0f;  // block 1: 4.0 * 0.5
  }

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<Float8E4M3FN>("data", {1, 32}, data, true);
  test.AddInput<int64_t>("indices", {1}, indices);
  test.AddInput<float>("scales", {1, 2}, scales);
  test.AddOutput<float>("output", {1, 32}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kFpExcludedProviders);
}

TEST(GatherBlockQuantizedOpTest, FpFloat16Output) {
  std::vector<Float8E4M3FN> data = {
      Float8E4M3FN(1.0f), Float8E4M3FN(2.0f),
      Float8E4M3FN(4.0f), Float8E4M3FN(8.0f)};
  std::vector<MLFloat16> scales = {MLFloat16(1.0f), MLFloat16(2.0f)};  // shape [2, 1]
  std::vector<int32_t> indices = {0, 1};
  std::vector<MLFloat16> expected = {
      MLFloat16(1.0f), MLFloat16(2.0f),
      MLFloat16(8.0f), MLFloat16(16.0f)};

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {2, 2}, data, true);
  test.AddInput<int32_t>("indices", {2}, indices);
  test.AddInput<MLFloat16>("scales", {2, 1}, scales);
  test.AddOutput<MLFloat16>("output", {2, 2}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kFpExcludedProviders);
}

#ifdef USE_CUDA
TEST(GatherBlockQuantizedOpTest, HostPageablePolicySelection) {
  using contrib::cuda::GatherBlockQuantizedDataPolicy;
  using contrib::cuda::SelectGatherBlockQuantizedDataPolicy;

  EXPECT_EQ(SelectGatherBlockQuantizedDataPolicy(false, true, true, true, true),
            GatherBlockQuantizedDataPolicy::DeviceCopy);
  EXPECT_EQ(SelectGatherBlockQuantizedDataPolicy(false, false, false, true, true),
            GatherBlockQuantizedDataPolicy::DeviceCopy);
  EXPECT_EQ(SelectGatherBlockQuantizedDataPolicy(true, true, true, true, true),
            GatherBlockQuantizedDataPolicy::DirectHost);
  EXPECT_EQ(SelectGatherBlockQuantizedDataPolicy(true, true, false, true, true),
            GatherBlockQuantizedDataPolicy::DeviceCopy);
  EXPECT_EQ(SelectGatherBlockQuantizedDataPolicy(true, true, true, false, true),
            GatherBlockQuantizedDataPolicy::DeviceCopy);
  EXPECT_EQ(SelectGatherBlockQuantizedDataPolicy(true, true, true, true, false),
            GatherBlockQuantizedDataPolicy::DeviceCopy);
}

#ifndef BUILD_CUDA_EP_AS_PLUGIN
TEST(GatherBlockQuantizedOpTest, HostPageableProviderOptionRoundTripAndHash) {
  CUDAExecutionProviderInfo default_info =
      CUDAExecutionProviderInfo::FromProviderOptions({});
  EXPECT_FALSE(default_info.enable_host_pageable_gather);
  EXPECT_EQ(OrtCUDAProviderOptionsV2{}.enable_host_pageable_gather, 0);

  CUDAExecutionProviderInfo disabled_info =
      CUDAExecutionProviderInfo::FromProviderOptions({{"enable_host_pageable_gather", "0"}});
  EXPECT_FALSE(disabled_info.enable_host_pageable_gather);

  CUDAExecutionProviderInfo enabled_info =
      CUDAExecutionProviderInfo::FromProviderOptions({{"enable_host_pageable_gather", "1"}});
  EXPECT_TRUE(enabled_info.enable_host_pageable_gather);
  const ProviderOptions serialized = CUDAExecutionProviderInfo::ToProviderOptions(enabled_info);
  ASSERT_EQ(serialized.count("enable_host_pageable_gather"), 1u);
  EXPECT_EQ(serialized.at("enable_host_pageable_gather"), "1");
  EXPECT_TRUE(CUDAExecutionProviderInfo::FromProviderOptions(serialized).enable_host_pageable_gather);
  EXPECT_NE(std::hash<CUDAExecutionProviderInfo>{}(disabled_info),
            std::hash<CUDAExecutionProviderInfo>{}(enabled_info));
}

#endif

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(GatherBlockQuantizedOpTest, FpFallbackWithPrepackingDisabledCuda) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "CUDA not available";
  }

  OrtCUDAProviderOptionsV2 info;
  info.enable_host_pageable_gather = 0;
  auto cuda_ep = CudaExecutionProviderWithOptions(&info);
  if (cuda_ep == nullptr) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {2, 2},
                              {Float8E4M3FN(1.0f), Float8E4M3FN(2.0f),
                               Float8E4M3FN(3.0f), Float8E4M3FN(4.0f)},
                              true);
  test.AddInput<int32_t>("indices", {2}, {1, 0});
  test.AddInput<float>("scales", {2, 1}, {2.0f, 0.5f});
  test.AddOutput<float>("output", {2, 2}, {1.5f, 2.0f, 2.0f, 4.0f});

  SessionOptions session_options;
  ASSERT_STATUS_OK(
      session_options.config_options.AddConfigEntry(kOrtSessionOptionsConfigDisablePrepacking, "1"));
  test.Config(session_options);

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(cuda_ep));
  test.ConfigEps(std::move(providers));
  test.RunWithConfig();
}

TEST(GatherBlockQuantizedOpTest, FpDirectHostPageableCuda) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "CUDA not available";
  }

  int pageable_memory_access = 0;
  int uses_host_page_tables = 0;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10020
  if (cudaDeviceGetAttribute(&pageable_memory_access, cudaDevAttrPageableMemoryAccess, 0) != cudaSuccess ||
      cudaDeviceGetAttribute(&uses_host_page_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0) !=
          cudaSuccess) {
    cudaGetLastError();
    GTEST_SKIP() << "CUDA pageable-memory attributes are unavailable";
  }
#endif
  if (pageable_memory_access == 0 || uses_host_page_tables == 0) {
    GTEST_SKIP() << "CUDA device does not use host page tables for pageable memory";
  }

  OrtCUDAProviderOptionsV2 info;
  info.enable_host_pageable_gather = 1;
  auto cuda_ep = CudaExecutionProviderWithOptions(&info);
  if (cuda_ep == nullptr) {
    GTEST_SKIP() << "CUDA EP not available";
  }

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {2, 2},
                              {Float8E4M3FN(1.0f), Float8E4M3FN(2.0f),
                               Float8E4M3FN(3.0f), Float8E4M3FN(4.0f)},
                              true);
  test.AddInput<int64_t>("indices", {3}, {-1, 0, 1});
  test.AddInput<MLFloat16>("scales", {2, 1}, {MLFloat16(2.0f), MLFloat16(0.5f)});
  test.AddOutput<MLFloat16>("output", {3, 2},
                            {MLFloat16(1.5f), MLFloat16(2.0f),
                             MLFloat16(2.0f), MLFloat16(4.0f),
                             MLFloat16(1.5f), MLFloat16(2.0f)});

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(cuda_ep));
  test.ConfigEps(std::move(providers));
  test.RunWithConfig();
}

TEST(GatherBlockQuantizedOpTest, FpDirectHostPageableCudaGraph) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "CUDA not available";
  }

  int pageable_memory_access = 0;
  int uses_host_page_tables = 0;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 10020
  if (cudaDeviceGetAttribute(&pageable_memory_access, cudaDevAttrPageableMemoryAccess, 0) != cudaSuccess ||
      cudaDeviceGetAttribute(&uses_host_page_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0) !=
          cudaSuccess) {
    cudaGetLastError();
    GTEST_SKIP() << "CUDA pageable-memory attributes are unavailable";
  }
#endif
  if (pageable_memory_access == 0 || uses_host_page_tables == 0) {
    GTEST_SKIP() << "CUDA device does not use host page tables for pageable memory";
  }

  OrtCUDAProviderOptionsV2 info;
  info.enable_cuda_graph = 1;
  info.enable_host_pageable_gather = 1;
  auto cuda_ep = CudaExecutionProviderWithOptions(&info);
  if (cuda_ep == nullptr) {
    GTEST_SKIP() << "CUDA EP not available";
  }
  IExecutionProvider* cuda_ep_ptr = cuda_ep.get();

  const std::vector<Float8E4M3FN> data = {
      Float8E4M3FN(1.0f), Float8E4M3FN(2.0f),
      Float8E4M3FN(3.0f), Float8E4M3FN(4.0f),
      Float8E4M3FN(5.0f), Float8E4M3FN(6.0f),
      Float8E4M3FN(7.0f), Float8E4M3FN(8.0f)};
  const size_t data_bytes = data.size() * sizeof(data[0]);
  const auto temp_dir_path =
      std::filesystem::temp_directory_path() /
      ("ort_gather_block_quantized_cuda_graph_" +
       std::to_string(reinterpret_cast<uintptr_t>(&pageable_memory_access)));
  TemporaryDirectory temp_dir(temp_dir_path.native());
  const auto data_path = temp_dir_path / "data.bin";
  {
    std::ofstream data_file(data_path, std::ios::binary);
    ASSERT_TRUE(data_file.good());
    data_file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data_bytes));
    ASSERT_TRUE(data_file.good());
  }

  Env::MappedMemoryPtr mapped_memory;
  ASSERT_STATUS_OK(Env::Default().MapFileIntoMemory(data_path.c_str(), 0, data_bytes, mapped_memory));
  const void* const mapped_address = mapped_memory.get();
  OrtMemoryInfo cpu_memory_info{CPU, OrtDeviceAllocator};
  Tensor mapped_tensor(DataTypeImpl::GetType<Float8E4M3FN>(), TensorShape({4, 2}),
                       mapped_memory.get(), cpu_memory_info);
  OrtValue mapped_data_value;
  Tensor::InitOrtValue(std::move(mapped_tensor), mapped_data_value);

  std::unordered_map<std::string, int> domain_to_version = {{onnxruntime::kMSDomain, 1}};
  std::vector<ONNX_NAMESPACE::FunctionProto> model_specific_functions;
  auto model = std::make_unique<Model>(
      "gather_block_quantized_cuda_graph", true, ModelMetaData(), PathString(),
      IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, model_specific_functions,
      DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
  auto& graph = model->MainGraph();

  std::vector<ONNX_NAMESPACE::TypeProto> tensor_types;
  tensor_types.reserve(4);
  auto add_tensor_type = [&](int elem_type, std::initializer_list<int64_t> dims) {
    tensor_types.emplace_back();
    auto* type = &tensor_types.back();
    type->mutable_tensor_type()->set_elem_type(elem_type);
    auto* shape = type->mutable_tensor_type()->mutable_shape();
    for (const int64_t dim : dims) {
      shape->add_dim()->set_dim_value(dim);
    }
    return type;
  };

  auto& data_arg = graph.GetOrCreateNodeArg(
      "data", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN, {4, 2}));
  auto& indices_arg = graph.GetOrCreateNodeArg(
      "indices", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_INT64, {2}));
  auto& scales_arg = graph.GetOrCreateNodeArg(
      "scales", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {4, 1}));
  auto& output_arg = graph.GetOrCreateNodeArg(
      "output", add_tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 2}));

  ONNX_NAMESPACE::TensorProto data_initializer;
  data_initializer.set_name("data");
  data_initializer.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN);
  data_initializer.add_dims(4);
  data_initializer.add_dims(2);
  data_initializer.mutable_raw_data()->assign(data_bytes, '\0');
  graph.AddInitializedTensor(data_initializer);

  NodeAttributes attributes = {
      {"block_size", utils::MakeAttribute("block_size", int64_t{0})},
      {"gather_axis", utils::MakeAttribute("gather_axis", int64_t{0})},
      {"quantize_axis", utils::MakeAttribute("quantize_axis", int64_t{1})},
  };
  auto& node = graph.AddNode("gather_block_quantized", "GatherBlockQuantized",
                             "CUDA Graph direct host-pageable test",
                             {&data_arg, &indices_arg, &scales_arg}, {&output_arg},
                             &attributes, onnxruntime::kMSDomain);
  node.SetExecutionProviderType(cuda_ep_ptr->Type());
  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_string;
  ASSERT_TRUE(model->ToProto().SerializeToString(&model_string));
  std::stringstream model_stream(model_string);

  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.AddInitializer("data", &mapped_data_value));
  {
    InferenceSession session(session_options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda_ep)));
    auto device_allocators = cuda_ep_ptr->CreatePreferredAllocators();
    const OrtMemoryInfo* device_memory_info = nullptr;
    for (const auto& allocator : device_allocators) {
      if (allocator->Info().device.Type() == OrtDevice::GPU &&
          allocator->Info().mem_type == OrtMemTypeDefault) {
        device_memory_info = &allocator->Info();
        break;
      }
    }
    ASSERT_NE(device_memory_info, nullptr);
    ASSERT_STATUS_OK(session.Load(model_stream));
    ASSERT_STATUS_OK(session.Initialize());
    auto device_allocator = session.GetAllocator(*device_memory_info);
    ASSERT_NE(device_allocator, nullptr);

    auto make_gpu_value = [&](const auto& values, const TensorShape& shape) {
      using T = typename std::decay_t<decltype(values)>::value_type;
      Tensor cpu_tensor(DataTypeImpl::GetType<T>(), shape, const_cast<T*>(values.data()), cpu_memory_info);
      Tensor gpu_tensor(DataTypeImpl::GetType<T>(), shape, device_allocator);
      ORT_THROW_IF_ERROR(cuda_ep_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
      OrtValue value;
      Tensor::InitOrtValue(std::move(gpu_tensor), value);
      return value;
    };

    std::vector<int64_t> indices = {0, 2};
    const std::vector<float> scales = {1.0f, 0.5f, 2.0f, 0.25f};
    auto indices_value = make_gpu_value(indices, TensorShape({2}));
    auto scales_value = make_gpu_value(scales, TensorShape({4, 1}));
    auto output_value = make_gpu_value(std::vector<float>(4), TensorShape({2, 2}));

    std::unique_ptr<IOBinding> io_binding;
    ASSERT_STATUS_OK(session.NewIOBinding(&io_binding));
    ASSERT_STATUS_OK(io_binding->BindInput("indices", indices_value));
    ASSERT_STATUS_OK(io_binding->BindInput("scales", scales_value));
    ASSERT_STATUS_OK(io_binding->BindOutput("output", output_value));

    RunOptions run_options;
    ASSERT_STATUS_OK(run_options.config_options.AddConfigEntry("gpu_graph_id", "1"));
    for (int i = 0; i < 3; ++i) {
      ASSERT_STATUS_OK(session.Run(run_options, *io_binding));
    }
    ASSERT_TRUE(cuda_ep_ptr->IsGraphCaptured(1));

    auto verify_output = [&](std::initializer_list<float> expected) {
      ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
      std::vector<float> actual(expected.size());
      Tensor cpu_output(DataTypeImpl::GetType<float>(), TensorShape({2, 2}), actual.data(), cpu_memory_info);
      ASSERT_STATUS_OK(cuda_ep_ptr->GetDataTransfer()->CopyTensor(output_value.Get<Tensor>(), cpu_output));
      EXPECT_EQ(actual, std::vector<float>(expected.begin(), expected.end()));
    };
    verify_output({1.0f, 2.0f, 10.0f, 12.0f});

    indices = {3, 1};
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(indices_value.GetMutable<Tensor>()->MutableData<int64_t>(), indices.data(),
                         indices.size() * sizeof(indices[0]), cudaMemcpyHostToDevice));
    ASSERT_STATUS_OK(session.Run(run_options, *io_binding));
    verify_output({1.75f, 2.0f, 1.5f, 2.0f});

#ifndef _WIN32
    ASSERT_EQ(0, madvise(mapped_memory.get(), data_bytes, MADV_DONTNEED));
    ASSERT_STATUS_OK(session.Run(run_options, *io_binding));
    verify_output({1.75f, 2.0f, 1.5f, 2.0f});
#endif

    EXPECT_EQ(mapped_address, mapped_memory.get());
  }
}
#endif

TEST(GatherBlockQuantizedOpTest, FpBFloat16OutputCuda) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "CUDA not available";
  }

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {1, 2},
                              {Float8E4M3FN(1.0f), Float8E4M3FN(2.0f)});
  test.AddInput<int64_t>("indices", {1}, {0});
  test.AddInput<BFloat16>("scales", {1, 1}, {BFloat16(2.0f)});
  test.AddOutput<BFloat16>("output", {1, 2}, {BFloat16(2.0f), BFloat16(4.0f)});

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(GatherBlockQuantizedOpTest, FpNegativeAxesCuda) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "CUDA not available";
  }

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", -2);
  test.AddAttribute<int64_t>("quantize_axis", -1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {1, 2},
                              {Float8E4M3FN(1.0f), Float8E4M3FN(2.0f)});
  test.AddInput<int64_t>("indices", {1}, {0});
  test.AddInput<float>("scales", {1, 1}, {2.0f});
  test.AddOutput<float>("output", {1, 2}, {2.0f, 4.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(GatherBlockQuantizedOpTest, FpEmptyIndicesCuda) {
  if (!HasCudaEnvironment(0)) {
    GTEST_SKIP() << "CUDA not available";
  }

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {1, 2},
                              {Float8E4M3FN(1.0f), Float8E4M3FN(2.0f)});
  test.AddInput<int64_t>("indices", {0}, {});
  test.AddInput<float>("scales", {1, 1}, {1.0f});
  test.AddOutput<float>("output", {0, 2}, {});

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}
#endif

TEST(GatherBlockQuantizedOpTest, FpInvalidBlockSizeThrows) {
  std::vector<Float8E4M3FN> data = {Float8E4M3FN(1.0f), Float8E4M3FN(2.0f)};
  std::vector<float> scales = {1.0f};
  std::vector<int64_t> indices = {0};

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 8);  // not a power of 2 >= 16, and not 0
  test.AddInput<Float8E4M3FN>("data", {1, 2}, data, true);
  test.AddInput<int64_t>("indices", {1}, indices);
  test.AddInput<float>("scales", {1, 1}, scales);
  test.AddOutput<float>("output", {1, 2}, {1.0f, 2.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "block_size must be a power of 2",
           kFpExcludedProviders);
}

TEST(GatherBlockQuantizedOpTest, FpRank3NonLeadingGatherAxisDifferentQuantizeAxis) {
  // data: [2, 3, 4] FP8 E4M3FN, all elements = 1.0. gather_axis = 1 (non-leading), quantize_axis = 2.
  // scales: [2, 3, 1], one scale per (outer, row) pair, distinct across both the leading axis (0,
  // untouched by gather) and the gathered axis (1), so that a wrong axis-stride computation would
  // be caught by mismatched expected values.
  std::vector<Float8E4M3FN> data(24, Float8E4M3FN(1.0f));
  std::vector<float> scales = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // shape [2, 3, 1]
  std::vector<int64_t> indices = {0, 2};
  std::vector<float> expected = {
      1.0f, 1.0f, 1.0f, 1.0f, 3.0f, 3.0f, 3.0f, 3.0f,
      4.0f, 4.0f, 4.0f, 4.0f, 6.0f, 6.0f, 6.0f, 6.0f};

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 1);
  test.AddAttribute<int64_t>("quantize_axis", 2);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {2, 3, 4}, data, true);
  test.AddInput<int64_t>("indices", {2}, indices);
  test.AddInput<float>("scales", {2, 3, 1}, scales);
  test.AddOutput<float>("output", {2, 2, 4}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kFpExcludedProviders);
}

TEST(GatherBlockQuantizedOpTest, FpValidNegativeIndices) {
  // Same table as FpBasicPerRowScale, but indices are negative (Python-style, relative to
  // gather_axis's dim size of 4): -3 == 1, -1 == 3.
  std::vector<Float8E4M3FN> data = {
      Float8E4M3FN(1.0f), Float8E4M3FN(2.0f), Float8E4M3FN(4.0f), Float8E4M3FN(8.0f),
      Float8E4M3FN(-1.0f), Float8E4M3FN(-2.0f), Float8E4M3FN(-4.0f), Float8E4M3FN(-8.0f),
      Float8E4M3FN(1.0f), Float8E4M3FN(1.0f), Float8E4M3FN(1.0f), Float8E4M3FN(1.0f),
      Float8E4M3FN(2.0f), Float8E4M3FN(2.0f), Float8E4M3FN(2.0f), Float8E4M3FN(2.0f)};
  std::vector<float> scales = {1.0f, 0.5f, 2.0f, 3.0f};  // shape [4, 1]
  std::vector<int64_t> indices = {-3, -1};
  std::vector<float> expected = {
      -0.5f, -1.0f, -2.0f, -4.0f,
      6.0f, 6.0f, 6.0f, 6.0f};

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {4, 4}, data, true);
  test.AddInput<int64_t>("indices", {2}, indices);
  test.AddInput<float>("scales", {4, 1}, scales);
  test.AddOutput<float>("output", {2, 4}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kFpExcludedProviders);
}

TEST(GatherBlockQuantizedOpTest, FpInvalidOutOfRangeIndexZeroFills) {
  std::vector<Float8E4M3FN> data = {
      Float8E4M3FN(1.0f), Float8E4M3FN(2.0f), Float8E4M3FN(4.0f), Float8E4M3FN(8.0f),
      Float8E4M3FN(-1.0f), Float8E4M3FN(-2.0f), Float8E4M3FN(-4.0f), Float8E4M3FN(-8.0f),
      Float8E4M3FN(1.0f), Float8E4M3FN(1.0f), Float8E4M3FN(1.0f), Float8E4M3FN(1.0f),
      Float8E4M3FN(2.0f), Float8E4M3FN(2.0f), Float8E4M3FN(2.0f), Float8E4M3FN(2.0f)};
  std::vector<float> scales = {1.0f, 0.5f, 2.0f, 3.0f};  // shape [4, 1]
  std::vector<int64_t> indices = {4};                    // out of range for a dim of size 4 ([-4, 3])

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {4, 4}, data, true);
  test.AddInput<int64_t>("indices", {1}, indices);
  test.AddInput<float>("scales", {4, 1}, scales);
  test.AddOutput<float>("output", {1, 4}, {0.0f, 0.0f, 0.0f, 0.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kFpExcludedProviders);
}
#endif  // !defined(DISABLE_FLOAT8_TYPES)

#if !defined(DISABLE_FLOAT4_TYPES)
TEST(GatherBlockQuantizedOpTest, Fp4BasicPerRowScale) {
  // data: [2, 4] FP4 E2M1, packed 2 logical elements per byte (logical shape is unaffected by packing,
  // same convention as the existing UInt4x2/Int4x2 sub-byte tensor types).
  // row0 = [1, 2, 4, 6], row1 = [-1, -2, -4, -6]; block_size = 0 -> one scale per row.
  std::vector<Float4E2M1x2> data = {
      Float4E2M1x2(1.0f, 2.0f), Float4E2M1x2(4.0f, 6.0f),
      Float4E2M1x2(-1.0f, -2.0f), Float4E2M1x2(-4.0f, -6.0f)};
  std::vector<float> scales = {1.0f, 0.5f};  // shape [2, 1]
  std::vector<int64_t> indices = {0, 1};
  std::vector<float> expected = {
      1.0f, 2.0f, 4.0f, 6.0f,
      -0.5f, -1.0f, -2.0f, -3.0f};

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float4E2M1x2>("data", {2, 4}, data, true);
  test.AddInput<int64_t>("indices", {2}, indices);
  test.AddInput<float>("scales", {2, 1}, scales);
  test.AddOutput<float>("output", {2, 4}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kFpExcludedProviders);
}

TEST(GatherBlockQuantizedOpTest, Fp4OddLogicalDimension) {
  // data: [1, 5] FP4 E2M1: an odd logical quantize-axis dimension, so the last packed byte holds
  // only one meaningful nibble (the second nibble of the final Float4E2M1x2 element is padding).
  std::vector<Float4E2M1x2> data = {
      Float4E2M1x2(1.0f, 2.0f), Float4E2M1x2(4.0f, 6.0f), Float4E2M1x2(-1.0f, 0.0f)};
  std::vector<float> scales = {1.0f};  // shape [1, 1]: block_size = 0 -> one scale for the whole row
  std::vector<int64_t> indices = {0};
  std::vector<float> expected = {1.0f, 2.0f, 4.0f, 6.0f, -1.0f};

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float4E2M1x2>("data", {1, 5}, data, true);
  test.AddInput<int64_t>("indices", {1}, indices);
  test.AddInput<float>("scales", {1, 1}, scales);
  test.AddOutput<float>("output", {1, 5}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kFpExcludedProviders);
}

TEST(GatherBlockQuantizedOpTest, Fp4OddRowsStartOnHighNibble) {
  std::vector<Float4E2M1x2> data = {
      Float4E2M1x2(1.0f, 2.0f), Float4E2M1x2(4.0f, 6.0f),
      Float4E2M1x2(-1.0f, -2.0f), Float4E2M1x2(-4.0f, -6.0f),
      Float4E2M1x2(0.5f, 1.0f), Float4E2M1x2(2.0f, 4.0f),
      Float4E2M1x2(6.0f, -0.5f), Float4E2M1x2(0.0f, 0.0f)};

  OpTester test("GatherBlockQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float4E2M1x2>("data", {3, 5}, data, true);
  test.AddInput<int64_t>("indices", {1}, {1});
  test.AddInput<float>("scales", {3, 1}, {1.0f, 0.5f, 2.0f});
  test.AddOutput<float>("output", {1, 5}, {-1.0f, -2.0f, -3.0f, 0.25f, 0.5f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", kFpExcludedProviders);
}
#endif  // !defined(DISABLE_FLOAT4_TYPES)

}  // namespace test
}  // namespace onnxruntime
