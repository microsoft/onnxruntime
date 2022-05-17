// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) || defined(USE_ROCM)

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"

namespace onnxruntime {
namespace contrib {
namespace test {

namespace {

constexpr int64_t kMaskSeed = 42;
// If ratio is 0, it will go to inference mode even the training_mode is true.
const std::vector<float> kRatios{0.25f, 0.50f, 0.75f, 0.99f};
constexpr size_t kNumBitsPerElement = sizeof(uint32_t) * CHAR_BIT;

template <typename T>
void RunTestForInference(const std::vector<int64_t>& input_dims, bool has_ratio = false, bool has_training_mode = false,
                         bool has_mask = false) {
  size_t input_size =
      static_cast<size_t>(std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>()));
  std::vector<T> input_data = onnxruntime::test::ValueRange<T>(input_size, T(1.f), T(1.f));
  onnxruntime::test::OpTester test("BitmaskDropout", 1, kMSDomain);
  test.AddInput<T>("data", input_dims, input_data);
  if (has_ratio) {
    // Ratio is ignored in inference mode.
    test.AddInput<float>("ratio", {}, {0.25f});
  } else if (has_training_mode) {
    test.AddOptionalInputEdge<float>();
  }

  if (has_training_mode) {
    test.AddInput<bool>("training_mode", {}, {false});
  }

  test.AddOutput<T>("output", input_dims, input_data);
  if (has_mask) {
    size_t mask_size = (input_size + kNumBitsPerElement - 1) / kNumBitsPerElement;
    std::vector<uint32_t> mask_data(mask_size, 0xFFFFFFFF);
    test.AddOutput<uint32_t>("mask", {static_cast<int64_t>(mask_size)}, mask_data);
  }
  test.Run();
}

template <typename T>
void RunTestForInferenceWrapper() {
  RunTestForInference<T>({7, 9});
  RunTestForInference<T>({7, 9}, true);
  RunTestForInference<T>({2, 17}, false, true);
  RunTestForInference<T>({2, 17}, true, true);
  RunTestForInference<T>({2, 17}, true, true, true);

  // Vectorized.
  RunTestForInference<T>({4, 11});
  RunTestForInference<T>({4, 11}, true);
  RunTestForInference<T>({16, 16}, false, true);
  RunTestForInference<T>({16, 16}, true, true);
  RunTestForInference<T>({16, 16}, true, true, true);
}

std::vector<uint32_t> MasksToBitmasks(size_t size, const bool* mask_data) {
  std::vector<uint32_t> result;
  for (size_t i = 0; i < size; ++i) {
    size_t bitmask_idx = i / kNumBitsPerElement;
    size_t bitmask_shift = i % kNumBitsPerElement;
    if (bitmask_idx >= result.size()) {
      result.push_back(0);
    }

    if (mask_data[i]) {
      result[bitmask_idx] |= (1 << bitmask_shift);
    }
  }

  return result;
}

template <typename T>
void RunTestForTraining(const std::vector<int64_t>& input_dims) {
  size_t input_size =
      static_cast<size_t>(std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>()));
  std::vector<T> input_data = onnxruntime::test::ValueRange<T>(input_size, T(0.f), T(1.f));
  for (float ratio : kRatios) {
    onnxruntime::test::OpTester dropout("Dropout", 13, kOnnxDomain, false);
    dropout.AddAttribute<int64_t>("seed", kMaskSeed);
    dropout.AddInput<T>("data", input_dims, input_data);
    dropout.AddInput<float>("ratio", {}, {ratio});
    dropout.AddInput<bool>("training_mode", {}, {true});

    // Outputs need to be populated with a value (even if incorrect), Test output won't actually be checked.
    // OpTester has no way to populate an output while explicitly ignoring its value.
    // We will fetch the outputs manually and use them to verify BitmaskDropout outputs.
    std::unique_ptr<bool[]> mask_buffer = std::make_unique<bool[]>(input_size);
    dropout.AddOutput<T>("output", input_dims, input_data);
    dropout.AddOutput<bool>("mask", input_dims, mask_buffer.get(), input_size);

    dropout.Run();
    std::vector<OrtValue> dropout_outputs = dropout.GetFetches();
    const T* output_values = dropout_outputs[0].Get<Tensor>().Data<T>();
    const bool* mask_values = dropout_outputs[1].Get<Tensor>().Data<bool>();
    std::vector<uint32_t> bitmask_values = MasksToBitmasks(input_size, mask_values);

    onnxruntime::test::OpTester bitmask_dropout("BitmaskDropout", 1, kMSDomain);
    bitmask_dropout.AddAttribute<int64_t>("seed", kMaskSeed);
    bitmask_dropout.AddInput<T>("data", input_dims, input_data);
    bitmask_dropout.AddInput<float>("ratio", {}, {ratio});
    bitmask_dropout.AddInput<bool>("training_mode", {}, {true});
    bitmask_dropout.AddOutput<T>("output", input_dims, output_values, input_size);
    bitmask_dropout.AddOutput<uint32_t>("mask", {static_cast<int64_t>(bitmask_values.size())}, bitmask_values);
    bitmask_dropout.Run();
  }
}

template <typename T>
void RunTestForTrainingWrapper() {
  RunTestForTraining<T>({7, 9});
  RunTestForTraining<T>({2, 17});

  // Vectorized.
  RunTestForTraining<T>({4, 11});
  RunTestForTraining<T>({16, 16});
}

}  // namespace

TEST(BitmaskDropoutTest, InferenceFloatType) { RunTestForInferenceWrapper<float>(); }

TEST(BitmaskDropoutTest, InferenceDoubleType) { RunTestForInferenceWrapper<double>(); }

TEST(BitmaskDropoutTest, InferenceHalfType) { RunTestForInferenceWrapper<MLFloat16>(); }

TEST(BitmaskDropoutTest, TrainingFloatType) { RunTestForTrainingWrapper<float>(); }

TEST(BitmaskDropoutTest, TrainingDoubleType) { RunTestForTrainingWrapper<double>(); }

TEST(BitmaskDropoutTest, TrainingHalfType) { RunTestForTrainingWrapper<MLFloat16>(); }

}  // namespace test
}  // namespace contrib
}  // namespace onnxruntime

#endif
