// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) || defined(USE_ROCM)

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#ifdef USE_ROCM
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#else
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#endif
#include "test/common/tensor_op_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

using BitmaskElementType = onnxruntime::cuda::BitmaskElementType;

void GetZeroPointEraseExpectedOutput(std::vector<float>& expected_output,
                                     std::vector<BitmaskElementType>& expected_bitmask_data,
                                     const std::vector<float>& input,
                                     const float zero_point) {
  for (size_t i = 0; i < input.size(); i += 4) {
    size_t bitmask_idx = i / onnxruntime::cuda::kNumBitsPerBitmaskElement;
    size_t bitmask_shift = i % onnxruntime::cuda::kNumBitsPerBitmaskElement;
    BitmaskElementType thread_bitmask = 0;
    for (int j = 0; j < 4; ++j) {
      if (i + j < input.size()) {
        if (input[i + j] != zero_point) {
          expected_output.push_back(input[i + j]);
          thread_bitmask |= (1 << j);
        }
      }
    }

    expected_bitmask_data[bitmask_idx] |= (thread_bitmask << bitmask_shift);
  }
}

template <typename T>
void RunZeroPointEraseTest(const std::vector<T>& input_data,
                           const std::vector<int64_t>& input_shape,
                           const float zero_point_value,
                           const std::vector<T>& expected_output,
                           const std::vector<BitmaskElementType>& expected_bitmask_data) {
  OpTester test("ZeroPointErase", 1, onnxruntime::kMSDomain);
  test.AddAttribute("zero_point", zero_point_value);

  test.AddInput<T>("input", input_shape, input_data);
  test.AddOutput<T>("output", std::vector<int64_t>{static_cast<int64_t>(expected_output.size())}, expected_output);
  test.AddOutput<BitmaskElementType>("mask",
                                     std::vector<int64_t>{static_cast<int64_t>(expected_bitmask_data.size())},
                                     expected_bitmask_data);
  test.AddOutput<int64_t>("input_shape",
                          std::vector<int64_t>{static_cast<int64_t>(input_shape.size())},
                          input_shape);
  test.Run();
}

TEST(ZeroPointEraseAndRestoreTest, EraseFloat) {
  std::vector<float> input_data{1.0f, 2.0f, 3.0f, 0.0f, 0.01f, 0.02f, 4.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f};
  std::vector<int64_t> input_shape{3, 4};
  std::vector<float> expected_output;
  expected_output.reserve(input_data.size());
  float zero_point_value = 0.0f;
  std::vector<BitmaskElementType> expected_bitmask_data;

  size_t bitmask_elem_count = (input_data.size() + onnxruntime::cuda::kNumBitsPerBitmaskElement - 1) /
                              onnxruntime::cuda::kNumBitsPerBitmaskElement;
  expected_bitmask_data.resize(bitmask_elem_count);

  GetZeroPointEraseExpectedOutput(expected_output, expected_bitmask_data, input_data, zero_point_value);

  RunZeroPointEraseTest(input_data, input_shape, zero_point_value, expected_output, expected_bitmask_data);
}

TEST(ZeroPointEraseAndRestoreTest, EraseFloat16) {
  std::vector<float> input_data{1.0f, 2.0f, 3.0f, 0.0f, 0.01f, 0.02f, 4.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f};
  std::vector<int64_t> input_shape{3, 4};
  std::vector<float> expected_output;
  expected_output.reserve(input_data.size());
  float zero_point_value = 0.0f;
  std::vector<BitmaskElementType> expected_bitmask_data;

  size_t bitmask_elem_count = (input_data.size() + onnxruntime::cuda::kNumBitsPerBitmaskElement - 1) /
                              onnxruntime::cuda::kNumBitsPerBitmaskElement;
  expected_bitmask_data.resize(bitmask_elem_count);
  GetZeroPointEraseExpectedOutput(expected_output, expected_bitmask_data, input_data, zero_point_value);

  RunZeroPointEraseTest(ToFloat16(input_data), input_shape, zero_point_value, ToFloat16(expected_output),
                        expected_bitmask_data);
}

TEST(ZeroPointEraseAndRestoreTest, EraseFloatNonDefaultZeroPointValue) {
  std::vector<float> input_data{1.0f, 2.0f, 3.0f, 0.0f, 0.01f, 0.02f, 1.0f, 0.0f, 0.0f, 1.0f, 6.0f, 1.0f};
  std::vector<int64_t> input_shape{3, 4};
  std::vector<float> expected_output;
  expected_output.reserve(input_data.size());
  float zero_point_value = 1.0f;
  std::vector<BitmaskElementType> expected_bitmask_data;

  size_t bitmask_elem_count = (input_data.size() + onnxruntime::cuda::kNumBitsPerBitmaskElement - 1) /
                              onnxruntime::cuda::kNumBitsPerBitmaskElement;
  expected_bitmask_data.resize(bitmask_elem_count);
  GetZeroPointEraseExpectedOutput(expected_output, expected_bitmask_data, input_data, zero_point_value);

  RunZeroPointEraseTest(input_data, input_shape, zero_point_value, expected_output, expected_bitmask_data);
}

void GetZeroPointRestoreExpectedOutput(size_t total_element_count,
                                       const std::vector<BitmaskElementType>& bitmask_data,
                                       const std::vector<float>& input,
                                       const float zero_point,
                                       std::vector<float>& expected_output) {
  expected_output.resize(total_element_count);
  size_t input_index = 0;
  for (size_t i = 0; i < total_element_count; i += 4) {
    size_t bitmask_idx = i / onnxruntime::cuda::kNumBitsPerBitmaskElement;
    size_t bitmask_shift = i % onnxruntime::cuda::kNumBitsPerBitmaskElement;
    float value_to_set;
    for (int j = 0; j < 4; ++j) {
      int mask_value = (1 << j) & bitmask_data[bitmask_idx] >> bitmask_shift;
      if (mask_value == 0) {
        value_to_set = zero_point;
      } else {
        value_to_set = input[input_index];
        input_index += 1;
      }

      expected_output[i + j] = value_to_set;
    }
  }
}

template <typename T>
void RunZeroPointRestoreTest(const std::vector<T>& input_data,
                             const std::vector<BitmaskElementType>& expected_bitmask_data,
                             const std::vector<int64_t>& output_shape,
                             const float zero_point_value,
                             const std::vector<T>& expected_output) {
  OpTester test("ZeroPointRestore", 1, onnxruntime::kMSDomain);
  test.AddAttribute("zero_point", zero_point_value);

  test.AddInput<T>("input", std::vector<int64_t>{static_cast<int64_t>(input_data.size())}, input_data);
  test.AddInput<BitmaskElementType>("mask",
                                    std::vector<int64_t>{static_cast<int64_t>(expected_bitmask_data.size())},
                                    expected_bitmask_data);
  test.AddInput<int64_t>("output_shape",
                         std::vector<int64_t>{static_cast<int64_t>(output_shape.size())},
                         output_shape);
  test.AddOutput<T>("output", output_shape, expected_output);
  test.Run();
}

TEST(ZeroPointEraseAndRestoreTest, RestoreFloat) {
  std::vector<int64_t> output_shape{3, 4};
  std::vector<float> expected_output;
  size_t total_element_count = 12;
  expected_output.reserve(total_element_count);
  float zero_point_value = 0.0f;
  // 12 elements will use one single bitmask element.
  // 32 bits: 0000,0000,0000,0000,0000,0001,1001,1111
  // The last 1111 means, element 0, 1, 2, 3 are not zero point.
  // The second last 1001 means, element 4, 7 are zero point, element 5, 6 are non zero point.
  constexpr std::bitset<32> bitmask_vector{0B00000000000000000000000110011111};
  std::vector<BitmaskElementType> bitmask_input_data{static_cast<BitmaskElementType>(bitmask_vector.to_ulong())};

  std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  GetZeroPointRestoreExpectedOutput(total_element_count, bitmask_input_data, input_data, zero_point_value,
                                    expected_output);

  RunZeroPointRestoreTest(input_data, bitmask_input_data, output_shape, zero_point_value, expected_output);
}

TEST(ZeroPointEraseAndRestoreTest, RestoreFloat16) {
  std::vector<int64_t> output_shape{3, 4};
  std::vector<float> expected_output;
  size_t total_element_count = 12;
  expected_output.reserve(total_element_count);
  float zero_point_value = 0.0f;
  // 12 elements will use one single bitmask element.
  // 32 bits: 0000,0000,0000,0000,0000,0001,1001,1111
  // The last 1111 means, element 0, 1, 2, 3 are not zero point.
  // The second last 1001 means, element 4, 7 are zero point, element 5, 6 are non zero point.
  constexpr std::bitset<32> bitmask_vector{0B00000000000000000000000110011111};
  std::vector<BitmaskElementType> bitmask_input_data{static_cast<BitmaskElementType>(bitmask_vector.to_ulong())};

  std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  GetZeroPointRestoreExpectedOutput(total_element_count, bitmask_input_data, input_data, zero_point_value,
                                    expected_output);

  RunZeroPointRestoreTest(ToFloat16(input_data), bitmask_input_data, output_shape, zero_point_value,
                          ToFloat16(expected_output));
}

TEST(ZeroPointEraseAndRestoreTest, RestoreFloatNonDefaultZeroPointValue) {
  std::vector<int64_t> output_shape{3, 4};
  std::vector<float> expected_output;
  size_t total_element_count = 12;
  expected_output.reserve(total_element_count);
  float zero_point_value = 23.0f;
  // 12 elements will use one single bitmask element.
  // 32 bits: 0000,0000,0000,0000,0000,0001,1001,1111
  // The last 1111 means, element 0, 1, 2, 3 are not zero point.
  // The second last 1001 means, element 4, 7 are zero point, element 5, 6 are non zero point.
  constexpr std::bitset<32> bitmask_vector{0B00000000000000000000000110011111};
  std::vector<BitmaskElementType> bitmask_input_data{static_cast<BitmaskElementType>(bitmask_vector.to_ulong())};

  std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  GetZeroPointRestoreExpectedOutput(total_element_count, bitmask_input_data, input_data, zero_point_value,
                                    expected_output);

  RunZeroPointRestoreTest(input_data, bitmask_input_data, output_shape, zero_point_value,
                          expected_output);
}

}  // namespace test
}  // namespace onnxruntime

#endif
