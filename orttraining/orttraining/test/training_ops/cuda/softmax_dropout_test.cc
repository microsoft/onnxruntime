// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) || defined(USE_ROCM)

#include <ctime>
#include <cstdlib>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

const std::vector<float> kRatios{-1.0f, 0.0f, 0.25f, 0.5f, 0.75f, 0.9f};

template <typename T>
void LaunchBiasSoftmaxDropoutTester(const std::vector<int64_t>& input_dims, const std::vector<T>& input_data,
                                    const std::vector<int64_t>& bias_dims, const std::vector<T>& bias_data,
                                    const std::vector<float>& softmax_output_data, int64_t axis,
                                    bool is_inner_broadcast, float ratio, float abs_error = 0.005f) {
  OpTester tester("BiasSoftmaxDropout", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("axis", axis);
  tester.AddAttribute<int64_t>("is_inner_broadcast", static_cast<int64_t>(is_inner_broadcast ? 1 : 0));
  tester.AddAttribute<int64_t>("seed", 42LL);
  tester.AddInput<T>("data", input_dims, input_data);
  tester.AddInput<T>("bias", bias_dims, bias_data);
  if (ratio == -1.0f) {
    tester.AddOptionalInputEdge<T>();
    ratio = 0.5f;
  } else {
    tester.AddInput<T>("ratio", {}, {T(ratio)});
  }

  // We'll do our own output verification so the output data here is meaningless.
  tester.AddOutput<T>("dropout_output", input_dims, input_data);
  size_t input_size = input_data.size();
  std::unique_ptr<bool[]> mask_buffer = std::make_unique<bool[]>(input_size);
  tester.AddOutput<bool>("mask", input_dims, mask_buffer.get(), input_size);
  tester.AddOutput<T>("softmax_output", input_dims, input_data);

  auto output_verifier = [&](const std::vector<OrtValue>& fetches, const std::string& provider_type) {
    ASSERT_EQ(fetches.size(), 3);
    const auto& dropout_output_tensor = fetches[0].Get<Tensor>();
    auto dropout_output_span = dropout_output_tensor.DataAsSpan<T>();
    const auto& mask_tensor = fetches[1].Get<Tensor>();
    auto mask_span = mask_tensor.DataAsSpan<bool>();
    const auto& softmax_output_tensor = fetches[2].Get<Tensor>();
    auto softmax_output_span = softmax_output_tensor.DataAsSpan<T>();

    const auto num_dropped_values = std::count(mask_span.begin(), mask_span.end(), false);
    ASSERT_NEAR(static_cast<float>(num_dropped_values) / static_cast<float>(mask_span.size()), ratio, 0.05f)
        << "provider: " << provider_type;

    ASSERT_EQ(mask_span.size(), dropout_output_span.size()) << "provider: " << provider_type;
    ASSERT_EQ(mask_span.size(), softmax_output_span.size()) << "provider: " << provider_type;
    for (size_t i = 0; i < mask_span.size(); ++i) {
      ASSERT_NEAR(static_cast<float>(softmax_output_span[i]), softmax_output_data[i], abs_error)
          << "provider: " << provider_type;
      if (mask_span[i]) {
        ASSERT_NEAR(static_cast<float>(dropout_output_span[i]), softmax_output_data[i] / (1.0f - ratio), abs_error)
            << "provider: " << provider_type;
      } else {
        ASSERT_EQ(dropout_output_span[i], T(0.0f)) << "provider: " << provider_type;
      }
    }
  };

  tester.SetCustomOutputVerifier(output_verifier);
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
#ifdef USE_CUDA
  eps.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  eps.push_back(DefaultRocmExecutionProvider());
#endif
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

void RunBiasSoftmaxDropoutTestInternal(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& bias_dims,
                                       int64_t axis, bool is_inner_broadcast, float ratio) {
  size_t new_axis = static_cast<size_t>(axis < 0 ? axis + static_cast<int64_t>(input_dims.size()) : axis);
  size_t element_count = static_cast<size_t>(
      std::accumulate(input_dims.cbegin() + new_axis, input_dims.cend(), 1LL, std::multiplies<int64_t>()));
  size_t batch_count = static_cast<size_t>(
      std::accumulate(input_dims.cbegin(), input_dims.cbegin() + new_axis, 1LL, std::multiplies<int64_t>()));
  size_t input_size = batch_count * element_count;
  RandomValueGenerator random{2333};
  std::vector<float> input_data = random.Uniform<float>(input_dims, -1.0f, 1.0f);
  std::vector<float> bias_data = random.Uniform<float>(bias_dims, 0.0f, 1.0f);
  std::vector<float> output_data(input_size);
  size_t rank = input_dims.size();
  size_t offset = input_dims.size() - bias_dims.size();
  std::vector<int64_t> input_strides(rank);
  std::vector<int64_t> bias_strides(rank);
  input_strides[rank - 1] = bias_strides[rank - 1] = 1;
  if (rank > 1) {
    int64_t bias_stride = bias_dims[rank - 1 - offset];
    for (size_t i = rank - 2;; --i) {
      input_strides[i] = input_dims[i + 1] * input_strides[i + 1];
      bias_strides[i] = i < offset || input_dims[i] != bias_dims[i - offset] ? 0 : bias_stride;
      if (i >= offset) bias_stride *= bias_dims[i - offset];
      if (i == 0) break;
    }
  }

  for (int64_t i = 0; i < static_cast<int64_t>(input_size); ++i) {
    int64_t bias_offset = 0;
    int64_t remain = i;
    for (size_t j = 0; j < rank; ++j) {
      int64_t q = remain / input_strides[j];
      bias_offset += q * bias_strides[j];
      remain = remain % input_strides[j];
    }
    output_data[i] = input_data[i] + bias_data[bias_offset];
  }

  for (size_t batch = 0; batch < batch_count; ++batch) {
    auto start = output_data.begin() + batch * element_count;
    auto end = start + element_count;
    float max = *std::max_element(start, end);
    float sum = std::accumulate(start, end, 0.0f, [max](float sum, float x) { return sum + exp(x - max); });
    std::transform(start, end, start, [max, sum](float x) { return exp(x - max) / sum; });
  }

  // For float.
  LaunchBiasSoftmaxDropoutTester(input_dims, input_data, bias_dims, bias_data, output_data, axis, is_inner_broadcast,
                                 ratio);

  // For fp16.
  LaunchBiasSoftmaxDropoutTester(input_dims, ToFloat16(input_data), bias_dims, ToFloat16(bias_data), output_data, axis,
                                 is_inner_broadcast, ratio, 0.05f);
}

void RunBiasSoftmaxDropoutTest(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& bias_dims,
                               int64_t axis, bool is_inner_broadcast) {
  for (float ratio : kRatios) {
    RunBiasSoftmaxDropoutTestInternal(input_dims, bias_dims, axis, is_inner_broadcast, ratio);
  }
}

template <typename T>
void LaunchSoftmaxDropoutGradTester(const std::vector<int64_t>& dims, const std::vector<T>& dy_data,
                                    const bool* mask_data, const std::vector<T>& y_data,
                                    const std::vector<float>& dx_data, int64_t axis, float ratio,
                                    float abs_error = .005f) {
  OpTester test("SoftmaxDropoutGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddInput<T>("dy", dims, dy_data);
  test.AddInput<bool>("mask", dims, mask_data, dy_data.size());
  test.AddInput<T>("softmax_y", dims, y_data);
  if (ratio == -1.0f) {
    test.AddOptionalInputEdge<T>();
  } else {
    test.AddInput<T>("ratio", {}, {static_cast<T>(ratio)});
  }

  // We'll do our own output verification so the output data here is meaningless.
  test.AddOutput<T>("dx", dims, dy_data);

  auto output_verifier = [&](const std::vector<OrtValue>& fetches, const std::string& provider_type) {
    ASSERT_EQ(fetches.size(), 1);
    const auto& dx_tensor = fetches[0].Get<Tensor>();
    auto dx_span = dx_tensor.DataAsSpan<T>();
    ASSERT_EQ(dx_data.size(), dx_span.size()) << "provider: " << provider_type;
    for (size_t i = 0; i < dx_data.size(); ++i) {
      ASSERT_NEAR(static_cast<float>(dx_span[i]), dx_data[i], abs_error) << "provider: " << provider_type;
    }
  };

  test.SetCustomOutputVerifier(output_verifier);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

void RunSoftmaxDropoutGradTestInternal(const std::vector<int64_t>& dims, int64_t axis, float ratio) {
  size_t new_axis = static_cast<size_t>(axis < 0 ? axis + static_cast<int64_t>(dims.size()) : axis);
  size_t element_count =
      static_cast<size_t>(std::accumulate(dims.cbegin() + new_axis, dims.cend(), 1LL, std::multiplies<int64_t>()));
  size_t batch_count =
      static_cast<size_t>(std::accumulate(dims.cbegin(), dims.cbegin() + new_axis, 1LL, std::multiplies<int64_t>()));
  size_t input_size = batch_count * element_count;
  RandomValueGenerator random{2333};
  std::vector<float> dy_data = random.Uniform<float>(dims, -0.1f, 0.1f);
  std::vector<float> y_data = random.Uniform<float>(dims, 0.0f, 1.0f);
  std::vector<float> dx_data(input_size);

  // DropoutGrad.
  std::vector<float> dropout_dx_data(input_size);
  float actual_ratio = ratio == -1.0f ? 0.5f : ratio;
  const float scale = 1.0f / (1.0f - actual_ratio);
  std::unique_ptr<bool[]> mask_buffer = std::make_unique<bool[]>(input_size);
  bool* mask_data = mask_buffer.get();
  int threshold = static_cast<int>(actual_ratio * 100);
  std::srand(static_cast<unsigned>(std::time(0)));
  for (size_t i = 0; i < input_size; ++i) {
    mask_data[i] = (std::rand() % 100) >= threshold ? true : false;
    dropout_dx_data[i] = dy_data[i] * static_cast<float>(mask_data[i]) * scale;
  }

  // SoftmaxGrad: dX = Y * ( dY - dot(Y, dY)) = Y * ( dY - ReduceSum(Y * dY))
  for (size_t i = 0; i < input_size; ++i) dx_data[i] = dropout_dx_data[i] * y_data[i];
  for (size_t batch = 0; batch < batch_count; ++batch) {
    size_t idx = batch * element_count;
    float sum = std::accumulate(dx_data.begin() + idx, dx_data.begin() + idx + element_count, 0.0f);
    for (size_t element = 0; element < element_count; ++element)
      dx_data[idx + element] = y_data[idx + element] * (dropout_dx_data[idx + element] - sum);
  }

  // For float.
  LaunchSoftmaxDropoutGradTester(dims, dy_data, mask_data, y_data, dx_data, axis, ratio);

  // For fp16.
  LaunchSoftmaxDropoutGradTester(dims, ToFloat16(dy_data), mask_data, ToFloat16(y_data), dx_data, axis, ratio, 0.05f);
}

void RunSoftmaxDropoutGradTest(const std::vector<int64_t>& dims, int64_t axis) {
  for (float ratio : kRatios) {
    RunSoftmaxDropoutGradTestInternal(dims, axis, ratio);
  }
}

}  // namespace

TEST(BiasSoftmaxDropoutTest, InnerBroadcast) {
  RunBiasSoftmaxDropoutTest({8, 4, 4, 2, 2, 8}, {8, 4, 1, 1, 2, 8}, 4, true);
}

TEST(BiasSoftmaxDropoutTest, OuterBroadcast) {
  RunBiasSoftmaxDropoutTest({8, 4, 4, 2, 2, 32}, {1, 4, 2, 2, 32}, 4, false);
}

TEST(BiasSoftmaxDropoutTest, WrapwiseVectorized) { RunBiasSoftmaxDropoutTest({4, 2, 256}, {4, 1, 256}, 2, true); }

TEST(BiasSoftmaxDropoutTest, WrapwiseNonVectorized) { RunBiasSoftmaxDropoutTest({4, 2, 513}, {2, 513}, -1, false); }

// large softmax batch tests falls back to cuda DNN library
TEST(BiasSoftmaxDropoutTest, InnerBroadcastLargeBatch) {
  RunBiasSoftmaxDropoutTest({4, 2, 4096}, {4, 1, 4096}, 2, true);
}

TEST(BiasSoftmaxDropoutTest, OuterBroadcastLargeBatch) { RunBiasSoftmaxDropoutTest({4, 2, 4096}, {2, 4096}, 2, false); }

TEST(SoftmaxDropoutGradTest, SmallBatch) { RunSoftmaxDropoutGradTest({8, 4, 4, 2, 2, 8}, 4); }

TEST(SoftmaxDropoutGradTest, WrapwiseVectorized) { RunSoftmaxDropoutGradTest({4, 2, 256}, 2); }

TEST(SoftmaxDropoutGradTest, WrapwiseNonVectorized) { RunSoftmaxDropoutGradTest({4, 2, 513}, -1); }

// large softmax batch tests falls back to cuda DNN library
TEST(SoftmaxDropoutGradTest, LargeBatch) { RunSoftmaxDropoutGradTest({4, 2, 4096}, 2); }

}  // namespace test
}  // namespace onnxruntime

#endif
