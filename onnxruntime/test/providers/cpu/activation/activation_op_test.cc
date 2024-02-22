// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "activation_op_test.h"
#include "core/providers/cpu/activation/activations.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(ENABLE_TRAINING_OPS)
namespace {
void TestElementwiseGradientOp(
    const char* op,
    const std::vector<std::pair<std::string, std::vector<float>>>& inputs,
    std::function<float(const std::vector<float>&)> expected_func,
    const std::unordered_map<std::string, float> attrs = {},
    int opset_version = 7, const char* domain = kOnnxDomain) {
  const auto first_input = inputs.begin();
  ASSERT_NE(first_input, inputs.end());
  for (auto input = first_input; input != inputs.end(); ++input) {
    if (input == first_input) continue;
    ASSERT_EQ(first_input->second.size(), input->second.size());
  }

  OpTester test(op, opset_version, domain);

  for (auto attr : attrs) {
    test.AddAttribute(attr.first, attr.second);
  }

  const auto input_size = first_input->second.size();
  std::vector<int64_t> dims{static_cast<int64_t>(input_size)};

  std::vector<float> expected_vals;
  for (size_t i = 0; i < input_size; i++) {
    std::vector<float> params(inputs.size());
    std::transform(
        inputs.begin(), inputs.end(), params.begin(),
        [i](const std::pair<std::string, std::vector<float>>& input) {
          return input.second[i];
        });
    expected_vals.push_back(expected_func(params));
  }

  for (const auto& input : inputs) {
    test.AddInput<float>(input.first.c_str(), dims, input.second);
  }
  test.AddOutput<float>("dX", dims, expected_vals);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {});
}

constexpr float ReluGrad(float dy, float x) {
  return x > 0 ? dy : 0;
}

constexpr float SigmoidGrad(float dy, float y) {
  return dy * y * (1 - y);
}

constexpr float TanhGrad(float dy, float y) {
  return dy * (1 - y * y);
}

constexpr float LeakyReluGrad(float dy, float y, float alpha) {
  return dy * (y > 0.0f ? 1.0f : alpha);
}
}  // namespace
#endif

TEST_F(ActivationOpTest, Sigmoid) {
  TestActivationOp<float>("Sigmoid",
                          input_values,
                          [](float x) {
                            auto y = 1.f / (1.f + std::exp(-std::abs(x)));  // safe sigmoid
                            y = x > 0 ? y : 1 - y;
                            return y;
                          });
  TestActivationOp<double>("Sigmoid",
                           input_values_double,
                           [](double x) {
                             auto y = 1. / (1. + std::exp(-std::abs(x)));  // safe sigmoid
                             y = x > 0 ? y : 1 - y;
                             return y;
                           });
}

TEST_F(ActivationOpTest, HardSigmoid) {
  float alpha = 0.2f;
  float beta = 0.5f;
  TestActivationOp<float>("HardSigmoid",
                          input_values,
                          [alpha, beta](float x) {
                            return std::max(std::min((alpha * x + beta), 1.0f), 0.0f);
                          },
                          {{"alpha", alpha}, {"beta", beta}});
}

TEST_F(ActivationOpTest, Tanh) {
  TestActivationOp<float>("Tanh",
                          input_values,
                          [](float x) { return std::tanh(x); });
  TestActivationOp<double>("Tanh",
                           input_values_double,
                           [](double x) { return std::tanh(x); });
}

TEST_F(ActivationOpTest, Relu) {
  TestActivationOp<float>("Relu",
                          input_values,
                          [](float x) { return std::max(x, 0.0f); });
  TestActivationOp<double>(
      "Relu",
      input_values_double,
      [](double x) { return std::max(x, 0.0); },
      {}, {},
      /*is_tensorrt_supported=*/false);
  TestActivationOp<int8_t>(
      "Relu",
      input_values_int8,
      [](int8_t x) { return std::max(x, static_cast<int8_t>(0)); },
      {}, {},
      /*is_tensorrt_supported=*/false,
      /*opset_version= */ 14);
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
  TestActivationOp<MLFloat16>(
      "Relu",
      input_values_fp16,
      [](MLFloat16 x) {
        if (x.ToFloat() > 0.0f) return x;
        return MLFloat16();
      },
      {}, {},
      /*is_tensorrt_supported=*/false,
      /*opset_version= */ 11);
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
}

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST_F(ActivationOpTest, Sigmoid_fp16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  OpTester test("Sigmoid", 14);

  auto formula = [](float x) {
    auto y = 1.f / (1.f + std::exp(-std::abs(x)));  // safe sigmoid
    y = x > 0 ? y : 1 - y;
    return y;
  };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<MLFloat16> f_X(X.size());
  std::vector<MLFloat16> f_Y(Y.size());
  ConvertFloatToMLFloat16(X.data(), f_X.data(), static_cast<int>(X.size()));
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), static_cast<int>(Y.size()));

  test.AddInput<MLFloat16>("X", dims, f_X);
  test.AddOutput<MLFloat16>("Y", dims, f_Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST_F(ActivationOpTest, Tanh_fp16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  OpTester test("Tanh", 14);

  auto formula = [](float x) { return std::tanh(x); };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<MLFloat16> f_X(X.size());
  std::vector<MLFloat16> f_Y(Y.size());
  ConvertFloatToMLFloat16(X.data(), f_X.data(), static_cast<int>(X.size()));
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), static_cast<int>(Y.size()));

  test.AddInput<MLFloat16>("X", dims, f_X);
  test.AddOutput<MLFloat16>("Y", dims, f_Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST_F(ActivationOpTest, Relu_fp16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  OpTester test("Relu", 14);

  auto formula = [](float x) { return std::max(x, 0.0f); };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<MLFloat16> f_X(X.size());
  std::vector<MLFloat16> f_Y(Y.size());
  ConvertFloatToMLFloat16(X.data(), f_X.data(), static_cast<int>(X.size()));
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), static_cast<int>(Y.size()));

  test.AddInput<MLFloat16>("X", dims, f_X);
  test.AddOutput<MLFloat16>("Y", dims, f_Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
#endif

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DNNL)
TEST_F(ActivationOpTest, Sigmoid_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Sigmoid", 14);

  auto formula = [](float x) {
    auto y = 1.f / (1.f + std::exp(-std::abs(x)));  // safe sigmoid
    y = x > 0 ? y : 1 - y;
    return y;
  };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<BFloat16> bf_X = FloatsToBFloat16s(X);
  std::vector<BFloat16> bf_Y = FloatsToBFloat16s(Y);

  test.AddInput<BFloat16>("X", dims, bf_X);
  test.AddOutput<BFloat16>("Y", dims, bf_Y);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#elif USE_DNNL
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST_F(ActivationOpTest, Tanh_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Tanh", 14);

  auto formula = [](float x) { return std::tanh(x); };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<BFloat16> bf_X = FloatsToBFloat16s(X);
  std::vector<BFloat16> bf_Y = FloatsToBFloat16s(Y);

  test.AddInput<BFloat16>("X", dims, bf_X);
  test.AddOutput<BFloat16>("Y", dims, bf_Y);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#elif USE_DNNL
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST_F(ActivationOpTest, Relu_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Relu", 14);

  auto formula = [](float x) { return std::max(x, 0.0f); };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<BFloat16> bf_X = FloatsToBFloat16s(X);
  std::vector<BFloat16> bf_Y = FloatsToBFloat16s(Y);

  test.AddInput<BFloat16>("X", dims, bf_X);
  test.AddOutput<BFloat16>("Y", dims, bf_Y);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#elif USE_DNNL
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  // USE_CUDA || USE_ROCM || USE_DNNL

#if defined(USE_DNNL)
TEST_F(ActivationOpTest, LeakyRelu_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("LeakyRelu", 16);
  float alpha = 0.01f;  // oneDNN set alpha equal to 0.01
  auto formula = [alpha](float x) { return (x >= 0) ? x : alpha * x; };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<BFloat16> bf_X = FloatsToBFloat16s(X);
  std::vector<BFloat16> bf_Y = FloatsToBFloat16s(Y);

  test.AddInput<BFloat16>("X", dims, bf_X);
  test.AddOutput<BFloat16>("Y", dims, bf_Y);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_DNNL
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  // USE_DNNL

TEST_F(ActivationOpTest, Elu) {
  float alpha = 0.1f;
  TestActivationOp<float>("Elu",
                          input_values,
                          [alpha](float x) { return (x >= 0) ? x : alpha * (exp(x) - 1); },
                          {{"alpha", alpha}});
}

TEST_F(ActivationOpTest, Celu) {
  float alpha = -0.5f;
  TestActivationOp<float>(
      "Celu",
      input_values,
      // TODO: Investigate why gcc 4 fails to compile without the explicit cast
      [alpha](float x) { return std::max(0.0f, x) + std::min(0.0f, alpha * (static_cast<float>(exp(x / alpha)) - 1)); },
      // Disable on TensorRT as it seems like it doesn't yet support Celu
      {{"alpha", alpha}}, {}, false, 12);
}

TEST_F(ActivationOpTest, LeakyRelu) {
  float alpha = 0.1f;
  TestActivationOp<float>("LeakyRelu",
                          input_values,
                          [alpha](float x) { return (x >= 0) ? x : alpha * x; },
                          {{"alpha", alpha}}, {});
}

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
TEST_F(ActivationOpTest, LeakyRelu_fp16) {
  OpTester test("LeakyRelu", 11);
  float alpha = 0.01f;  // oneDNN set alpha equal to 0.01
  auto formula = [alpha](float x) { return (x >= 0) ? x : alpha * x; };

  std::vector<float> X = input_values.front();
  std::vector<float> Y;
  for (unsigned i = 0; i < X.size(); i++)
    Y.push_back(formula(X[i]));
  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<MLFloat16> bf_X(X.size());
  ConvertFloatToMLFloat16(X.data(), bf_X.data(), (int)X.size());
  std::vector<MLFloat16> bf_Y(Y.size());
  ConvertFloatToMLFloat16(Y.data(), bf_Y.data(), (int)Y.size());

  test.AddInput<MLFloat16>("X", dims, bf_X);
  test.AddOutput<MLFloat16>("Y", dims, bf_Y);
  test.Run();
}
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED

TEST_F(ActivationOpTest, ThresholdedRelu) {
  float alpha = 0.1f;
  TestActivationOp<float>(
      "ThresholdedRelu",
      input_values,
      [alpha](float x) { return (x >= alpha) ? x : 0; },
      {{"alpha", alpha}}, {}, true, 10);
}

TEST_F(ActivationOpTest, Selu) {
  static constexpr float alpha = 1.6732f;
  static constexpr float gamma = 1.0507f;

  TestActivationOp<float>("Selu",
                          input_values,
                          [](float x) { return x <= 0 ? gamma * (alpha * exp(x) - alpha) : gamma * x; },
                          {{"alpha", alpha}, {"gamma", gamma}}, {});
}

TEST_F(ActivationOpTest, Selu_Attributes) {
  static constexpr float alpha = 1.8f;
  static constexpr float gamma = 0.5f;

  TestActivationOp<float>("Selu",
                          input_values,
                          [](float x) { return x <= 0 ? gamma * (alpha * exp(x) - alpha) : gamma * x; },
                          {{"alpha", alpha}, {"gamma", gamma}}, {});
}

TEST_F(ActivationOpTest, Selu_GH10726) {
  static constexpr float alpha = -2.0f;
  static constexpr float gamma = -3.0f;

  TestActivationOp<float>("Selu",
                          {{1.f, -1.f}},
                          [](float x) { return x <= 0 ? gamma * (alpha * exp(x) - alpha) : gamma * x; },
                          {{"alpha", alpha}, {"gamma", gamma}}, {});
}

TEST_F(ActivationOpTest, PRelu) {
  OpTester test("PRelu");

  auto formula = [](float x, float slope) { return x < 0 ? slope * x : x; };

  std::vector<float> inputs{1.0f, -4.0f, 0.0f, -9.0f};
  std::vector<float> slopes{1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<float> outputs;
  for (unsigned i = 0; i < inputs.size(); i++)
    outputs.push_back(formula(inputs[i], slopes[i]));

  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, inputs);
  test.AddInput<float>("slope", dims, slopes);
  test.AddOutput<float>("Y", dims, outputs);
  test.Run();
}

TEST_F(ActivationOpTest, PRelu_SingleSlope) {
  auto test = [](bool slope_is_initializer) {
    SCOPED_TRACE(MakeString("slope_is_initializer: ", slope_is_initializer));

    OpTester test("PRelu");

    auto formula = [](float x, float slope) { return x < 0 ? slope * x : x; };

    auto inputs = {1.0f, 2.0f, -4.0f, 3.0f, 0.0f, 5.0f, -9.0f, 8.0f};
    auto slope = 1.5f;
    std::vector<float> outputs;
    for (auto& input : inputs)
      outputs.push_back(formula(input, slope));

    std::vector<int64_t> dims{2, 2, 2};
    test.AddInput<float>("X", dims, inputs);
    test.AddInput<float>("slope", {}, {slope}, slope_is_initializer);
    test.AddOutput<float>("Y", dims, outputs);
    test.Run();
  };

  test(true /* slope_is_initializer */);
  test(false /* slope_is_initializer */);
}

TEST_F(ActivationOpTest, PRelu_MultiChannel3D) {
  OpTester test("PRelu");

  auto formula = [](float x, float slope) { return x < 0 ? slope * x : x; };

  std::vector<float> inputs{1.0f, 2.0f, -4.0f, 3.0f, 0.0f, 5.0f, -9.0f, 8.0f};
  std::vector<float> slopes{1.0f, -2.0f};
  std::vector<float> outputs;
  constexpr int64_t num_images = 2;
  constexpr int64_t num_channels = 2;
  constexpr int64_t num_pixels = 2;
  for (unsigned i = 0; i < inputs.size(); i++)
    outputs.push_back(formula(inputs[i], slopes[i / num_pixels % num_channels]));

  std::vector<int64_t> x_dims{num_images, num_channels, num_pixels};
  std::vector<int64_t> slope_dims{num_channels, 1};
  test.AddInput<float>("X", x_dims, inputs);
  test.AddInput<float>("slope", slope_dims, slopes);
  test.AddOutput<float>("Y", x_dims, outputs);
  // QNN has some issue with the broadcast support
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
}

TEST_F(ActivationOpTest, PRelu_MultiChannel4D) {
  RandomValueGenerator random{2345};

  auto test = [&](bool slope_is_initializer,
                  int64_t n, int64_t c, int64_t h, int64_t w) {
    SCOPED_TRACE(MakeString("slope_is_initializer: ", slope_is_initializer,
                            ", n: ", n, ", c: ", c, ", h: ", h, ", w: ", w));

    OpTester test("PRelu");

    auto formula = [](float x, float slope) { return x < 0 ? slope * x : x; };

    const std::vector<int64_t> x_dims{n, c, h, w};
    const std::vector<int64_t> slope_dims{c, 1, 1};
    std::vector<float> inputs = random.Uniform<float>(x_dims, -16.0f, 16.0f);
    std::vector<float> slopes = random.Uniform<float>(slope_dims, -1.0f, 1.0f);
    std::vector<float> outputs;
    for (unsigned i = 0; i < inputs.size(); i++) {
      outputs.push_back(formula(inputs[i], slopes[i / (h * w) % c]));
    }

    test.AddInput<float>("X", x_dims, inputs);
    test.AddInput<float>("slope", slope_dims, slopes, slope_is_initializer);
    test.AddOutput<float>("Y", x_dims, outputs);
    // QNN has some issue with the broadcast support
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kQnnExecutionProvider});
  };

  test(true /* slope_is_initializer */, 5, 4, 3, 2);
  test(false, 5, 4, 3, 2);

  test(true, 3, 1, 1, 1);
  test(false, 3, 1, 1, 1);
}

TEST_F(ActivationOpTest, Softplus) {
  TestActivationOp<float>("Softplus",
                          input_values,
                          [](float x) {
                            if (x > 0)
                              return x + log1pf(expf(-x));
                            else
                              return log1pf(expf(x));
                          });
}

TEST_F(ActivationOpNoInfTest, Softsign) {
  if constexpr (!SessionOptions::DEFAULT_USE_PER_SESSION_THREADS) {
    GTEST_SKIP() << "Skipping the test";
  }
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 1, which exceeds threshold";
  }

  TestActivationOp<float>(
      "Softsign",
      input_values,
      [](float x) {
        auto result = x / (1 + std::abs(x));

#if defined(__arm__)
        // Softsign uses Eigen inverse(), which on ARM32 results in a different value when x is FLT_MAX or -FLT_MAX
        // 3.40282347e+38 -> 0 with ARM32 inverse() vs something like 2.939e-39#DEN with other platforms.
        //
        // Possibly explained by https://en.wikipedia.org/wiki/ARM_architecture#Advanced_SIMD_(Neon)
        // 'A quirk of Neon in Armv7 devices is that it flushes all subnormal numbers to zero'
        //
        // c.f.
        // cmake\external\eigen\Eigen\src\Core\arch\SSE\PacketMath.h uses _mm_div_ps for 'pdiv<Packet4f>'
        // cmake\external\eigen\Eigen\src\Core\arch\NEON\PacketMath.h uses a custom implementation for 'pdiv<Packet4f>'
        //
        // Special case the expected values to allow for that. If handling FLT_MAX more consistently is required
        // we'd need to not use Eigen for Softsign on ARM32.
        //
        if (x == FLT_MAX) {
          result = 0.;
        } else if (x == -FLT_MAX) {
          result = -0.;
        }
#endif

        return result;
      },
      {}, {}, false);  // Disable TensorRT because result mismatches
}

#if defined(ENABLE_TRAINING_OPS)
TEST(ReluGradInferenceTest, Basic) {
  const std::vector<float> x_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);

  TestElementwiseGradientOp(
      "ReluGrad",
      {{"dY", dY}, {"X", x_vals}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], x = params[1];

        return ReluGrad(dy, x);
      },
      {}, 1, kMSDomain);
}

TEST(SigmoidGradInferenceTest, Basic) {
  const std::vector<float> y_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);

  TestElementwiseGradientOp(
      "SigmoidGrad",
      {{"dY", dY}, {"Y", y_vals}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], y = params[1];

        return SigmoidGrad(dy, y);
      },
      {}, 1, kMSDomain);
}

TEST(TanhGradInferenceTest, Basic) {
  const std::vector<float> y_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);

  TestElementwiseGradientOp(
      "TanhGrad",
      {{"dY", dY}, {"Y", y_vals}},
      [](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], y = params[1];

        return TanhGrad(dy, y);
      },
      {}, 1, kMSDomain);
}

TEST(LeakyReluGradInferenceTest, Basic) {
  const std::vector<float> y_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  const std::vector<float> dY(7, 1.0f);
  float alpha = 0.5f;

  TestElementwiseGradientOp(
      "LeakyReluGrad",
      {{"dY", dY}, {"Y", y_vals}},
      [alpha](const std::vector<float>& params) {
        ORT_ENFORCE(params.size() == 2);
        const auto dy = params[0], y = params[1];

        return LeakyReluGrad(dy, y, alpha);
      },
      {{"alpha", alpha}}, 1, kMSDomain);
}
#endif

// Remove DNNL from running this test because DNNL Gelu op seems not check domain for kernel implementation.
// It will run the DNNL Gelu op which only be part of standard of Gelu-20 op.
#if !defined(USE_DNNL) && !defined(USE_QNN)
TEST_F(ActivationOpTest, ONNX_Gelu) {
  TestActivationOp<float>(
      "Gelu",
      input_values,
      [](float x) { return 0.5 * x * (1 + erf(x * M_SQRT1_2)); }, {},
      {{"approximate", "none"}}, true, 20);

  TestActivationOp<float>(
      "Gelu",
      input_values,
      [](float x) { return 0.5 * x * (1 + erf(x * M_SQRT1_2)); },
      {},
      {/*default value of approximate attribute is none */}, true, 20);

  TestActivationOp<float>(
      "Gelu",
      input_values,
      [](float x) {
        return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * x * x * x)));
      },
      {},
      {{"approximate", "tanh"}}, true, 20);
}
#endif

}  // namespace test
}  // namespace onnxruntime
