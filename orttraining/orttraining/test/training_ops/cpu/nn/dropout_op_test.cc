// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4389)
#endif

#include "orttraining/training_ops/cpu/nn/dropout_op.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <random>

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace contrib {
namespace test {

using namespace onnxruntime::test;

namespace {
constexpr auto k_dropout_opset_version = 12;

const Tensor& FetchTensor(const OrtValue& ort_value) {
  if (ort_value.Fence()) {
    ort_value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, 0);
  }
  return ort_value.Get<Tensor>();
}

void RunDropoutTest(const char* op, const bool use_mask, const std::vector<int64_t>& input_shape, float ratio = -1,
                    bool training_mode = true, bool use_float16_ratio = false) {
  OpTester t{op, k_dropout_opset_version, kOnnxDomain};

  const auto input_size = std::accumulate(
      input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<>{});
  std::vector<float> input(input_size);
  std::iota(input.begin(), input.end(), 1.0f);
  const int64_t seed = 42;

  t.AddAttribute("seed", seed);
  t.AddInput("data", input_shape, input);
  if (ratio == -1) {
    ratio = 0.5;  // default.
    t.AddInput("ratio", {}, {ratio});
  } else if (use_float16_ratio) {
    t.AddInput("ratio", {}, {MLFloat16(0)});
  } else {
    t.AddInput("ratio", {}, {ratio});
  }

  if (strcmp(op, "TrainableDropout") != 0 && training_mode) {
    t.AddInput("training_mode", {}, {true});
  }

  t.AddOutput<float>("output", input_shape, input);  // we'll do our own output verification

  std::unique_ptr<bool[]> mask_buffer{};
  if (use_mask) {
    mask_buffer = onnxruntime::make_unique<bool[]>(input_size);
    t.AddOutput<bool>("mask", input_shape, mask_buffer.get(), input_size);
  } else {
    t.AddMissingOptionalOutput<bool>();
  }

  auto output_verifier = [&](const std::vector<OrtValue>& fetches, const std::string& provider_type) {
    ASSERT_GE(fetches.size(), 1);
    const auto& output_tensor = FetchTensor(fetches[0]);
    auto output_span = output_tensor.DataAsSpan<float>();

    const auto num_output_zeros = std::count(output_span.begin(), output_span.end(), 0.0f);

    if (ratio == 1.0f) {
      ASSERT_EQ(num_output_zeros, static_cast<size_t>(output_span.size())) << "provider: " << provider_type;
    } else {
      ASSERT_NEAR(static_cast<float>(num_output_zeros) / static_cast<size_t>(output_span.size()), ratio, 0.1f)
          << "provider: " << provider_type;

      for (decltype(output_span.size()) i = 0; i < output_span.size(); ++i) {
        if (output_span[i] == 0.0f) continue;
        const auto expected_value = (i + 1.0f) / (1 - ratio);
        ASSERT_NEAR(output_span[i], expected_value, 0.01f)
            << "unexpected output value at index " << i << ", provider: " << provider_type;
      }
    }

    if (use_mask) {
      ASSERT_GE(fetches.size(), 2);
      const auto& mask_tensor = FetchTensor(fetches[1]);
      auto mask_span = mask_tensor.DataAsSpan<bool>();
      ASSERT_EQ(mask_span.size(), output_span.size()) << "provider: " << provider_type;

      const auto num_mask_zeros = std::count(mask_span.begin(), mask_span.end(), false);
      ASSERT_EQ(num_output_zeros, num_mask_zeros) << "provider: " << provider_type;

      for (decltype(mask_span.size()) i = 0; i < mask_span.size(); ++i) {
        ASSERT_TRUE(
            (mask_span[i] && output_span[i] != 0.0f) || (!mask_span[i] && output_span[i] == 0.0f))
            << "output and mask mismatch at index " << i << ", output[i]: " << output_span[i]
            << ", mask[i]: " << mask_span[i] << ", provider: " << provider_type;
      }
    }
  };

  t.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, nullptr, ExecutionMode::ORT_SEQUENTIAL, output_verifier);
}
}  // namespace

// Dropout

TEST(DropoutTest, Basic) {
  RunDropoutTest("Dropout", false, {10, 10, 10}, 0.75);
}

TEST(DropoutTest, Mask) {
  RunDropoutTest("Dropout", true, {1000}, 0.25);
}

TEST(DropoutTest, RatioLimit) {
  RunDropoutTest("Dropout", true, {1000}, 0.0f, false);
}

TEST(DropoutTest, EmptyRatio) {
  RunDropoutTest("Dropout", true, {1000});
}

TEST(DropoutTest, Float16Ratio) {
  RunDropoutTest("Dropout", true, {1000}, 0.0f, true, true);
}

TEST(TrainableDropoutTest, Basic) {
  RunDropoutTest("TrainableDropout", false, {10, 10, 10}, 0.75);
}

TEST(TrainableDropoutTest, Mask) {
  RunDropoutTest("TrainableDropout", true, {1000}, 0.25);
}

TEST(TrainableDropoutTest, RatioLimit) {
  RunDropoutTest("TrainableDropout", true, {1000}, 0.0f, false);
}

TEST(TrainableDropoutTest, EmptyRatio) {
  RunDropoutTest("TrainableDropout", true, {1000}, -1);
}

TEST(TrainableDropoutTest, Float16Ratio) {
  RunDropoutTest("TrainableDropout", true, {1000}, 0.0f, true, true);
}

namespace {
void RunDropoutGradTest(const char* op, float ratio, const std::vector<int64_t>& input_dims, bool default_ratio = true) {
  const auto input_shape = TensorShape(input_dims);
  OpTester test(op, 1, kMSDomain);
  if (default_ratio) {
    ratio = 0.5f;
  }
  const float input_constant = 3.0f;

  std::vector<float> dy_data(input_shape.Size(), input_constant);
  std::vector<float> ratio_data(1, ratio);

  auto mask_buffer = onnxruntime::make_unique<bool[]>(input_shape.Size());
  std::generate_n(
      mask_buffer.get(), input_shape.Size(),
      [ratio, rng = std::default_random_engine{42},
       dist = std::uniform_real_distribution<float>{0.0f, 1.0f}]() mutable {
        return dist(rng) >= ratio;
      });

  const float output_constant = input_constant / (1.0f - ratio);
  std::vector<float> dx_data{};
  dx_data.reserve(input_shape.Size());
  std::transform(
      mask_buffer.get(), mask_buffer.get() + input_shape.Size(), std::back_inserter(dx_data),
      [output_constant](bool mask_value) { return mask_value ? output_constant : 0.0f; });

  test.AddInput<float>("dy", input_shape.GetDims(), dy_data);
  test.AddInput<bool>("mask", input_shape.GetDims(), mask_buffer.get(), input_shape.Size());
  if (!default_ratio) {
    test.AddInput<float>("ratio", {1}, ratio_data);
  }

  test.AddOutput<float>("dx", input_shape.GetDims(), dx_data);

  test.Run();
}
}  // namespace

// DropoutGrad

TEST(DropoutGradTest, Basic) {
  //Ratio 0.2, 1D
  RunDropoutGradTest("DropoutGrad", 0.2f, {16}, false);

  //Ratio 0.3, 2D
  RunDropoutGradTest("DropoutGrad", 0.3f, {8, 2}, false);

  //Ratio 0.4, 3D
  RunDropoutGradTest("DropoutGrad", 0.4f, {2, 4, 2}, false);

  //default Ratio, 3D
  RunDropoutGradTest("DropoutGrad", 0.5f, {2, 4, 2});
}

TEST(DropoutGradTest, RatioLimit) {
  RunDropoutGradTest("DropoutGrad", 0.0f, {16}, false);
}

TEST(TrainableDropoutGradTest, Basic) {
  //Ratio 0.2, 1D
  RunDropoutGradTest("TrainableDropoutGrad", 0.2f, {16}, false);

  //Ratio 0.3, 2D
  RunDropoutGradTest("TrainableDropoutGrad", 0.3f, {8, 2}, false);

  //Ratio 0.4, 3D
  RunDropoutGradTest("TrainableDropoutGrad", 0.4f, {2, 4, 2}, false);

  //default Ratio, 3D
  RunDropoutGradTest("TrainableDropoutGrad", 0.5f, {2, 4, 2});
}

TEST(TrainableDropoutGradTest, RatioLimit) {
  RunDropoutGradTest("TrainableDropoutGrad", 0.0f, {16}, false);
}

}  // namespace test
}  // namespace contrib
}  // namespace onnxruntime
