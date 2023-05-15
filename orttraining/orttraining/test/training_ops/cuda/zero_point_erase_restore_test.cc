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

#include "core/util/math.h"

namespace onnxruntime {
namespace test {

using BitmaskElementType = onnxruntime::cuda::BitmaskElementType;

template <typename T>
void GetExpectedOutput(std::vector<T>& expected_output,
                       std::vector<BitmaskElementType>& expected_bitmask_data,
                       const std::vector<T>& input,
                       const float zero_point) {
  for (size_t i = 0; i < input.size(); ++i) {
    T value = input[i];
    size_t bitmask_idx = i / onnxruntime::cuda::kNumBitsPerBitmaskElement;
    size_t bitmask_shift = i % onnxruntime::cuda::kNumBitsPerBitmaskElement;
    float value_to_compare;
    if (std::is_same<T, MLFloat16>::value) {
      value_to_compare = math::halfToFloat(value);
    } else {
      value_to_compare = value;
    }

    if (value_to_compare != zero_point) {
      expected_output.push_back(value);
      expected_bitmask_data[bitmask_idx] |= (1 << bitmask_shift);
    }
  }
}

TEST(ZeroPointEraseAndRestoreTest, EraseFloat) {
  std::vector<float> input_data{1.0f, 2.0f, 3.0f, 0.0f, 0.01f, 0.02f, 4.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f};
  std::vector<int64_t> input_shape{3, 4};
  std::vector<float> expected_output;
  expected_output.reserve(input_data.size());
  float zero_point_value = 0.0f;
  std::vector<BitmaskElementType> expected_bitmask_data;
  expected_bitmask_data.resize((input_data.size() + onnxruntime::cuda::kNumBitsPerBitmaskElement - 1) / onnxruntime::cuda::kNumBitsPerBitmaskElement);
  GetExpectedOutput<float>(expected_output, expected_bitmask_data, input_data, zero_point_value);

  OpTester test("ZeroPointErase", 1, onnxruntime::kMSDomain);
  test.AddAttribute("zero_point", 0.0f);

  test.AddInput<float>("input", input_shape, input_data);
  test.AddOutput<float>("output", input_shape, expected_output);
  test.AddOutput<BitmaskElementType>("mask", std::vector<int64_t>{static_cast<int64_t>(expected_bitmask_data.size())},
                                     expected_bitmask_data);
  test.AddOutput<int64_t>("input_shape", std::vector<int64_t>{static_cast<int64_t>(input_shape.size())}, input_shape);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

#endif
