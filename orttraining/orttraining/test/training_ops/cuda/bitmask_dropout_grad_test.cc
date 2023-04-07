// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) || defined(USE_ROCM)

#include <ctime>
#include <cstdlib>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/default_providers.h"
#ifdef USE_ROCM
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#else
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#endif

namespace onnxruntime {
namespace contrib {
namespace test {

#ifdef USE_ROCM
using onnxruntime::rocm::BitmaskElementType;
using onnxruntime::rocm::kNumBitsPerBitmaskElement;
#else
using onnxruntime::cuda::BitmaskElementType;
using onnxruntime::cuda::kNumBitsPerBitmaskElement;
#endif

namespace {

const std::vector<float> kRatios{0.00f, 0.25f, 0.50f, 0.75f, 0.99f};

void GenerateMaskData(size_t size, bool* mask_data, float ratio) {
  int threshold = static_cast<int>(ratio * 100);
  std::srand(static_cast<unsigned>(std::time(0)));
  for (size_t i = 0; i < size; ++i) {
    mask_data[i] = (std::rand() % 100) >= threshold ? true : false;
  }
}

template <typename T>
int64_t GetDropoutGradBitmaskAndOutput(size_t size, const std::vector<T>& input_data, const bool* masks,
                                       const float ratio, std::vector<BitmaskElementType>& bitmask_data,
                                       std::vector<T>& output_data) {
  const float p = 1.0f - ratio;
  const float scale = 1.0f / p;
  bitmask_data.clear();
  output_data.resize(size);
  for (size_t i = 0; i < size; ++i) {
    output_data[i] = T(static_cast<float>(input_data[i]) * static_cast<float>(masks[i]) * scale);
    size_t bitmask_idx = i / kNumBitsPerBitmaskElement;
    size_t bitmask_shift = i % kNumBitsPerBitmaskElement;
    if (bitmask_idx >= bitmask_data.size()) {
      bitmask_data.push_back(0);
    }

    if (masks[i] == 1) {
      bitmask_data[bitmask_idx] |= (1 << bitmask_shift);
    }
  }

  return static_cast<int64_t>(bitmask_data.size());
}

template <typename T>
void RunTest(const std::vector<int64_t>& input_dims) {
  size_t input_size =
      static_cast<size_t>(std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>()));
  std::vector<T> input_data = onnxruntime::test::ValueRange<T>(input_size, T(1.f), T(1.f));
  std::unique_ptr<bool[]> mask_buffer = std::make_unique<bool[]>(input_size);
  for (float ratio : kRatios) {
    onnxruntime::test::OpTester test("BitmaskDropoutGrad", 1, kMSDomain);
    GenerateMaskData(input_size, mask_buffer.get(), ratio);
    std::vector<BitmaskElementType> bitmask_data;
    std::vector<T> output_data;
    int64_t bitmask_element_count =
        GetDropoutGradBitmaskAndOutput(input_size, input_data, mask_buffer.get(), ratio, bitmask_data, output_data);
    test.AddInput<T>("data", input_dims, input_data);
    test.AddInput<BitmaskElementType>("mask", {bitmask_element_count}, bitmask_data);
    test.AddInput<float>("ratio", {}, {ratio});
    test.AddInput<bool>("training_mode", {}, {true});
    test.AddOutput<T>("output", input_dims, output_data);
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
    execution_providers.push_back(onnxruntime::test::DefaultCudaExecutionProvider());
#elif USE_ROCM
    execution_providers.push_back(onnxruntime::test::DefaultRocmExecutionProvider());
#endif
    test.Run(onnxruntime::test::OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

template <typename T>
void RunTestWrapper() {
  RunTest<T>({7, 9});
  RunTest<T>({2, 17});

  // Vectorized.
  RunTest<T>({4, 11});
  RunTest<T>({16, 16});
}

}  // namespace

TEST(BitmaskDropoutGradTest, FloatType) { RunTestWrapper<float>(); }

TEST(BitmaskDropoutGradTest, DoubleType) { RunTestWrapper<double>(); }

TEST(BitmaskDropoutGradTest, HalfType) { RunTestWrapper<MLFloat16>(); }

}  // namespace test
}  // namespace contrib
}  // namespace onnxruntime

#endif
