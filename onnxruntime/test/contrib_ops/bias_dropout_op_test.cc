// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#pragma warning(disable : 4389)
#endif


#include <algorithm>
#include <memory>
#include <numeric>
#include <random>

#include "gtest/gtest.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace contrib {
namespace test {

using namespace onnxruntime::test;

enum TrainingMode { TrainingFalse, TrainingTrue, NoTraining };

// BiasDropout kernel is only implemented for CUDA/ROCM
#if defined(USE_CUDA) || defined(USE_ROCM)
namespace {
void RunBiasDropoutTest(const bool use_mask, const std::vector<int64_t>& input_shape, float ratio = -1.0f,
                        TrainingMode training_mode = TrainingTrue, bool use_float16_ratio = false, bool has_residual = true) {
  OpTester t{"BiasDropout", 1, kMSDomain};
  const int64_t seed = 42;
  t.AddAttribute("seed", seed);

  const auto input_size = std::accumulate(
      input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<>{});
  const std::vector<float> input = ValueRange(input_size, 1.0f, 1.0f);
  t.AddInput("data", input_shape, input);

  std::vector<int64_t> bias_shape{input_shape.back()};
  const auto bias_size = input_shape.back();
  const std::vector<float> bias = ValueRange(bias_size, 2.0f, 1.0f);
  t.AddInput("bias", bias_shape, bias);

  float residual_value = 0.0f;
  if (has_residual) {
    residual_value = 1.0f;
    const auto residual_size = input_size;
    const std::vector<float> residual(residual_size, residual_value);
    t.AddInput("residual", input_shape, residual);
  } else {
    t.AddMissingOptionalInput<float>();
  }

  if (ratio == -1.0f) {
    if (use_float16_ratio) {
      t.AddMissingOptionalInput<MLFloat16>();
    } else {
      t.AddMissingOptionalInput<float>();
    }
    // set ratio to default value
    ratio = 0.5f;
  } else {
    if (use_float16_ratio) {
      t.AddInput("ratio", {}, {MLFloat16(math::floatToHalf(ratio))});
    } else {
      t.AddInput("ratio", {}, {ratio});
    }
  }

  if (training_mode != NoTraining) {
    if (training_mode == TrainingTrue) {
      t.AddInput("training_mode", {}, {true});
    } else {
      t.AddInput("training_mode", {}, {false});
    }
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

    const auto num_dropped_values = std::count(output_span.begin(), output_span.end(), residual_value);

    if (ratio == 1.0f) {
      ASSERT_EQ(num_dropped_values, static_cast<size_t>(output_span.size())) << "provider: " << provider_type;
    } else {
      ASSERT_NEAR(static_cast<float>(num_dropped_values) / static_cast<size_t>(output_span.size()), training_mode == TrainingTrue ? ratio : 0.0f, 0.1f)
          << "provider: " << provider_type;

      for (decltype(output_span.size()) i = 0; i < output_span.size(); ++i) {
        if (output_span[i] == residual_value) continue;
        const auto expected_value = (bias[i % bias_size] + i + 1.0f) / (1 - (training_mode == TrainingTrue ? ratio : 0.0f)) + residual_value;
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
      ASSERT_EQ(num_dropped_values, num_mask_zeros) << "provider: " << provider_type;

      for (decltype(mask_span.size()) i = 0; i < mask_span.size(); ++i) {
        ASSERT_TRUE(
            (mask_span[i] && output_span[i] != residual_value) || (!mask_span[i] && output_span[i] == residual_value))
            << "output and mask mismatch at index " << i << ", output[i]: " << output_span[i]
            << ", mask[i]: " << mask_span[i] << ", provider: " << provider_type;
      }
    }
  };

  t.SetCustomOutputVerifier(output_verifier);

  t.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
}  // namespace

TEST(BiasDropoutTest, Basic) {
  RunBiasDropoutTest(false, {10, 10, 10}, 0.75f);
}
TEST(BiasDropoutTest, BasicTrainingFalse) {
  RunBiasDropoutTest(false, {10, 10, 10}, 0.75f, TrainingFalse);
}
TEST(BiasDropoutTest, BasicNoTraining) {
  RunBiasDropoutTest(false, {10, 10, 10}, 0.75f, NoTraining);
}

TEST(BiasDropoutTest, BasicWithoutResidual) {
  RunBiasDropoutTest(false, {10, 10, 10}, 0.75f, TrainingTrue, false, false);
}

TEST(BiasDropoutTest, Mask) {
  RunBiasDropoutTest(true, {3, 5, 768}, 0.25f);
}
TEST(BiasDropoutTest, MaskTrainingFalse) {
  RunBiasDropoutTest(true, {3, 5, 768}, 0.25f, TrainingFalse);
}
TEST(BiasDropoutTest, MaskNoTraining) {
  RunBiasDropoutTest(true, {3, 5, 768}, 0.25f, NoTraining);
}

TEST(BiasDropoutTest, RatioLimit) {
  RunBiasDropoutTest(true, {4, 8, 1024}, 0.0f, TrainingFalse);
}

TEST(BiasDropoutTest, EmptyRatio) {
  RunBiasDropoutTest(true, {2, 7, 1024});
}
#endif

}  // namespace test
}  // namespace contrib
}  // namespace onnxruntime
