// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include <algorithm>
#include <random>
using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {

TEST(Random, RandomNormal2DDouble) {
  OpTester test("RandomNormal");

  std::vector<int64_t> dims{20, 50};

  constexpr float scale = 10.f;
  constexpr float mean = 0.f;
  constexpr float seed = 123.f;

  test.AddAttribute("scale", scale);
  test.AddAttribute("mean", mean);
  test.AddAttribute("seed", seed);
  test.AddAttribute<int64_t>("dtype", TensorProto::DOUBLE);
  test.AddAttribute("shape", dims);

  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<double> distribution{mean, scale};

  std::vector<double> expected_output(TensorShape(dims).Size());
  std::for_each(expected_output.begin(), expected_output.end(),
                [&generator, &distribution](double& value) { value = distribution(generator); });

  test.AddOutput<double>("Y", dims, expected_output);

  // The expected_output is generated using std lib, which is used by CPU kernel only.
  // So we need to exclude other EPs here. Ditto for other places.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kRocmExecutionProvider});
}

void RunRandomNormalLike3DFloat(bool infer_dtype = false) {
  OpTester test("RandomNormalLike");

  std::vector<int64_t> dims{2, 2, 3};

  constexpr float scale = 10.f;
  constexpr float mean = 0.f;
  constexpr float seed = 123.f;

  test.AddAttribute("scale", scale);
  test.AddAttribute("mean", mean);
  test.AddAttribute("seed", seed);

  if (!infer_dtype)
    test.AddAttribute<int64_t>("dtype", TensorProto::FLOAT);

  test.AddInput<float>("X", dims,
                       {0.f, 0.f, 0.f,
                        0.f, 0.f, 0.f,

                        0.f, 0.f, 0.f,
                        0.f, 0.f, 0.f});

  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution{mean, scale};

  std::vector<float> expected_output(TensorShape(dims).Size());
  std::for_each(expected_output.begin(), expected_output.end(),
                [&generator, &distribution](float& value) { value = distribution(generator); });

  test.AddOutput<float>("Y", dims, expected_output);

  // TensorRT does not support manual seed overrides and there will be result mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kRocmExecutionProvider, kTensorrtExecutionProvider});
}

TEST(Random, RandomNormalLike3DDouble) {
  RunRandomNormalLike3DFloat();
}

TEST(Random, RandomNormalLikeInferDType) {
  constexpr bool infer_dtype = true;
  RunRandomNormalLike3DFloat(infer_dtype);
}

TEST(Random, RandomUniform1DFloat) {
  OpTester test("RandomUniform");

  std::vector<int64_t> dims{10};

  constexpr float low = 0.f;
  constexpr float high = 100.f;
  constexpr float seed = 123.f;

  test.AddAttribute("low", low);
  test.AddAttribute("high", high);
  test.AddAttribute("seed", seed);
  test.AddAttribute<int64_t>("dtype", TensorProto::FLOAT);
  test.AddAttribute("shape", dims);

  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::uniform_real_distribution<float> distribution{low, high};

  std::vector<float> expected_output(TensorShape(dims).Size());
  std::for_each(expected_output.begin(), expected_output.end(),
                [&generator, &distribution](float& value) { value = distribution(generator); });

  test.AddOutput<float>("Y", dims, expected_output);

  // TensorRT does not support manual seed overrides and there will be result mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kRocmExecutionProvider, kTensorrtExecutionProvider});
}

void RunRandomUniformLikeTest(bool infer_dtype = false) {
  OpTester test("RandomUniformLike");

  std::vector<int64_t> dims{2, 6};

  constexpr float low = 0.f;
  constexpr float high = 100.f;
  constexpr float seed = 123.f;

  test.AddAttribute("low", low);
  test.AddAttribute("high", high);
  test.AddAttribute("seed", seed);

  if (!infer_dtype)
    test.AddAttribute<int64_t>("dtype", TensorProto::DOUBLE);

  test.AddInput<double>("X", dims,
                        {0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0.});

  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::uniform_real_distribution<double> distribution{low, high};

  std::vector<double> expected_output(TensorShape(dims).Size());
  std::for_each(expected_output.begin(), expected_output.end(),
                [&generator, &distribution](double& value) { value = distribution(generator); });

  test.AddOutput<double>("Y", dims, expected_output);

  // TensorRT does not support seed parameter and there will be result mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kRocmExecutionProvider, kTensorrtExecutionProvider});
}

TEST(Random, RandomUniformLike2DDouble) {
  RunRandomUniformLikeTest();
}

TEST(Random, RandomUniformLikeInferDType) {
  constexpr bool infer_dtype = true;
  RunRandomUniformLikeTest(infer_dtype);
}

TEST(Random, InvalidDType) {
  constexpr float seed = 123.f;

  std::vector<int64_t> dims{1, 4};
  std::vector<int32_t> input{0, 0, 0, 0};
  std::vector<double> expected_output{0., 0., 0., 0.};

  {
    OpTester test("RandomNormal");

    float scale = 10.f;
    float mean = 0.f;

    test.AddAttribute("scale", scale);
    test.AddAttribute("mean", mean);
    test.AddAttribute("seed", seed);
    test.AddAttribute<int64_t>("dtype", 999);
    test.AddAttribute("shape", dims);

    test.AddOutput<double>("Y", dims, expected_output);
    test.Run(OpTester::ExpectResult::kExpectFailure, "Attribute dtype does not specify a valid type.");
  }

  {
    OpTester test("RandomUniform");

    constexpr float low = 0.f;
    constexpr float high = 100.f;

    test.AddAttribute("low", low);
    test.AddAttribute("high", high);
    test.AddAttribute("seed", seed);
    test.AddAttribute<int64_t>("dtype", 999);
    test.AddAttribute("shape", dims);

    test.AddOutput<double>("Y", dims, expected_output);
    test.Run(OpTester::ExpectResult::kExpectFailure, "Attribute dtype does not specify a valid type.");
  }

  {
    OpTester test("RandomNormalLike");

    constexpr float scale = 10.f;
    constexpr float mean = 0.f;

    test.AddAttribute("scale", scale);
    test.AddAttribute("mean", mean);
    test.AddAttribute("seed", seed);
    test.AddAttribute<int64_t>("dtype", 999);

    test.AddInput<int32_t>("X", dims, input);
    test.AddOutput<double>("Y", dims, expected_output);
    test.Run(OpTester::ExpectResult::kExpectFailure, "Attribute dtype does not specify a valid type.");
  }

  {
    OpTester test("RandomUniformLike");

    constexpr float low = 0.f;
    constexpr float high = 100.f;

    test.AddAttribute("low", low);
    test.AddAttribute("high", high);
    test.AddAttribute("seed", seed);
    test.AddAttribute<int64_t>("dtype", 999);

    test.AddInput<int32_t>("X", dims, input);
    test.AddOutput<double>("Y", dims, expected_output);
    test.Run(OpTester::ExpectResult::kExpectFailure, "Attribute dtype does not specify a valid type.");
  }
}

/*
Note: There are no reference tests that can be reused in this case. I tried to use the tensorflow
test cases but they use a different RNG (Philox) and hence the test results differ. Since the implementation
of the op is same as tensorflow, for now I've just relied on the output generated by this code as ground truth
for verification.
*/
TEST(Random, MultinomialGoodCase) {
  OpTester test("Multinomial");

  constexpr int64_t num_samples = 10;
  constexpr float seed = 1618.f;
  constexpr int batch_size = 2;
  constexpr int num_classes = 3;

  const std::vector<int64_t> input_dims{batch_size, num_classes};
  std::vector<float> input(TensorShape(input_dims).Size());
  std::fill(input.begin(), input.end(), -10.f);
  test.AddInput<float>("X", input_dims, input);

  test.AddAttribute("sample_size", num_samples);
  test.AddAttribute("seed", seed);
  test.AddAttribute<int64_t>("dtype", TensorProto::INT64);

  const std::vector<int64_t> output_dims{batch_size, num_samples};
#ifdef _WIN32
  const std::vector<int64_t> expected_output{2, 0, 0, 2, 2, 2, 0, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 0};
#elif defined(__MACH__) || defined(__ANDROID__) || defined(__FreeBSD__) || defined(__wasm__)
  const std::vector<int64_t> expected_output{1, 1, 2, 2, 0, 2, 2, 2, 0, 2, 1, 1, 2, 0, 2, 2, 0, 2, 1, 1};
#else
  const std::vector<int64_t> expected_output{2, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 1, 0, 1, 0, 2, 0, 2, 0};
#endif
  test.AddOutput<int64_t>("Y", output_dims, expected_output);

  test.Run();
}

TEST(Random, MultinomialDefaultDType) {
  auto run_test = [](int num_run_calls, const std::vector<int32_t>& expected_output) {
    OpTester test("Multinomial");
    constexpr int64_t num_samples = 10;
    constexpr int batch_size = 2;
    constexpr float seed = 1618.f;

    const std::vector<int64_t> input_dims{2, 3};
    std::vector<float> input(TensorShape(input_dims).Size());
    std::fill(input.begin(), input.end(), -10.f);
    test.AddInput<float>("X", input_dims, input);

    test.AddAttribute("sample_size", num_samples);
    test.AddAttribute("seed", seed);

    const std::vector<int64_t> output_dims{batch_size, num_samples};
    test.AddOutput<int32_t>("Y", output_dims, expected_output);

    // test.Run() re-loads the model each time, so we need to do multiple calls to InferenceSession::Run inside of it
    // to test that the second call to Compute produces different data
    test.SetNumRunCalls(num_run_calls);

    test.Run();
  };

#ifdef _WIN32
  const std::vector<int32_t> expected_output_1{2, 0, 0, 2, 2, 2, 0, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 0};
  const std::vector<int32_t> expected_output_2{0, 0, 1, 0, 2, 2, 2, 0, 2, 1, 2, 1, 0, 2, 0, 2, 2, 1, 2, 1};
#elif defined(__MACH__) || defined(__ANDROID__) || defined(__FreeBSD__) || defined(__wasm__)
  const std::vector<int32_t> expected_output_1{1, 1, 2, 2, 0, 2, 2, 2, 0, 2, 1, 1, 2, 0, 2, 2, 0, 2, 1, 1};
  const std::vector<int32_t> expected_output_2{1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 2, 0, 1, 1, 0, 2, 2, 2, 1};
#else
  const std::vector<int32_t> expected_output_1{2, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 1, 1, 0, 1, 0, 2, 0, 2, 0};
  const std::vector<int32_t> expected_output_2{2, 2, 1, 1, 0, 2, 2, 1, 1, 2, 0, 0, 0, 2, 0, 1, 1, 1, 0, 0};
#endif

  // Test output from a single call to Multinomial::Compute
  run_test(1, expected_output_1);

  // Test output from 2 calls to Multinomial::Compute
  run_test(2, expected_output_2);
}

TEST(Random, MultinomialInvalidDtype) {
  OpTester test("Multinomial");

  constexpr int64_t num_samples = 10;
  constexpr int batch_size = 2;
  constexpr int num_classes = 3;
  constexpr float seed = 1618.f;

  const std::vector<int64_t> input_dims{batch_size, num_classes};
  std::vector<float> input(TensorShape(input_dims).Size());
  std::fill(input.begin(), input.end(), -10.f);
  test.AddInput<float>("X", input_dims, input);

  test.AddAttribute("sample_size", num_samples);
  test.AddAttribute("seed", seed);
  test.AddAttribute<int64_t>("dtype", 999);

  const std::vector<int64_t> output_dims{batch_size, num_samples};
  const std::vector<int32_t> expected_output{2, 0, 0, 2, 2, 2, 0, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 0};
  test.AddOutput<int32_t>("Y", output_dims, expected_output);

  test.Run(OpTester::ExpectResult::kExpectFailure, "Output type must be int32 or int64");
}

#if defined(USE_CUDA) || defined(USE_ROCM)
// We cannot call CUDA lib from UT, so just do some simple verification on output tensor.
void RunRandomNormalGpuTest(const std::vector<int64_t> dims, const float mean, const float scale, const float seed,
                            TensorProto_DataType dtype, bool is_random_like, bool infer_dtype) {
  OpTester test(is_random_like ? "RandomNormalLike" : "RandomNormal");
  test.AddAttribute("mean", mean);
  test.AddAttribute("scale", scale);
  test.AddAttribute("seed", seed);
  if (!is_random_like) {
    test.AddAttribute<int64_t>("dtype", dtype);
  } else if (!infer_dtype) {
    // For RandomNormalLike, if not infer dtype, use float as target.
    test.AddAttribute<int64_t>("dtype", TensorProto_DataType::TensorProto_DataType_FLOAT);
  }
  size_t size = 1;
  for (size_t i = 0; i < dims.size(); ++i) {
    size *= static_cast<size_t>(dims[i]);
  }
  if (!is_random_like) {
    test.AddAttribute("shape", dims);
  } else {
    if (dtype == TensorProto_DataType::TensorProto_DataType_FLOAT) {
      std::vector<float> float_data(size, 0.f);
      test.AddInput("X", dims, float_data);
    } else if (dtype == TensorProto_DataType::TensorProto_DataType_DOUBLE) {
      std::vector<double> double_data(size, 0.);
      test.AddInput("X", dims, double_data);
    } else if (dtype == TensorProto_DataType::TensorProto_DataType_FLOAT16) {
      std::vector<float> float_data(size, 0.f);
      std::vector<MLFloat16> fp16_data(size);
      ConvertFloatToMLFloat16(float_data.data(), fp16_data.data(), static_cast<int>(size));
      test.AddInput("X", dims, fp16_data);
    }
  }

  // We'll do our own output verification.
  TensorProto_DataType output_dtype =
      is_random_like && !infer_dtype ? TensorProto_DataType::TensorProto_DataType_FLOAT : dtype;
  if (output_dtype == TensorProto_DataType::TensorProto_DataType_FLOAT) {
    std::vector<float> float_data(size, 0.f);
    test.AddOutput("Y", dims, float_data);
  } else if (output_dtype == TensorProto_DataType::TensorProto_DataType_DOUBLE) {
    std::vector<double> double_data(size, 0.);
    test.AddOutput("Y", dims, double_data);
  } else if (output_dtype == TensorProto_DataType::TensorProto_DataType_FLOAT16) {
    std::vector<float> float_data(size, 0.f);
    std::vector<MLFloat16> fp16_data(size);
    ConvertFloatToMLFloat16(float_data.data(), fp16_data.data(), static_cast<int>(size));
    test.AddOutput("Y", dims, fp16_data);
  }

  auto output_verifier = [&](const std::vector<OrtValue>& fetches, const std::string& provider_type) {
    // Only one output, and mean of output values are near attribute mean.
    ASSERT_EQ(fetches.size(), 1u);
    const auto& output_tensor = fetches[0].Get<Tensor>();
    if (output_dtype == TensorProto_DataType::TensorProto_DataType_FLOAT) {
      auto output_span = output_tensor.DataAsSpan<float>();
      float sum = std::accumulate(output_span.begin(), output_span.end(), 0.f);
      ASSERT_NEAR(sum / static_cast<float>(size), mean, 0.1f);
    } else if (output_dtype == TensorProto_DataType::TensorProto_DataType_DOUBLE) {
      auto output_span = output_tensor.DataAsSpan<double>();
      double sum = std::accumulate(output_span.begin(), output_span.end(), 0.);
      ASSERT_NEAR(sum / static_cast<double>(size), static_cast<double>(mean), 0.1);
    } else if (output_dtype == TensorProto_DataType::TensorProto_DataType_FLOAT16) {
      auto output_span = output_tensor.DataAsSpan<MLFloat16>();
      float sum = 0.f;
      for (auto value : output_span) {
        sum += value.ToFloat();
      }
      ASSERT_NEAR(sum / static_cast<float>(size), mean, 0.1f);
    }
  };

  test.SetCustomOutputVerifier(output_verifier);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider, kTensorrtExecutionProvider});
}

TEST(Random, RandomNormalGpu) {
  // We will call RandomVectorizedKernel if total_size % 4 == 0, so test two input sizes here.
  std::vector<int64_t> dims1{256, 256};
  RunRandomNormalGpuTest(dims1, 1.f, 10.f, 123.f, TensorProto_DataType::TensorProto_DataType_FLOAT, false, false);
  RunRandomNormalGpuTest(dims1, -1.f, 8.f, 231.f, TensorProto_DataType::TensorProto_DataType_DOUBLE, false, false);
  RunRandomNormalGpuTest(dims1, 0.f, 16.f, 312.f, TensorProto_DataType::TensorProto_DataType_FLOAT16, false, false);
  RunRandomNormalGpuTest(dims1, 1.f, 10.f, 123.f, TensorProto_DataType::TensorProto_DataType_FLOAT, true, true);
  RunRandomNormalGpuTest(dims1, -1.f, 8.f, 231.f, TensorProto_DataType::TensorProto_DataType_DOUBLE, true, true);
  RunRandomNormalGpuTest(dims1, 0.f, 16.f, 312.f, TensorProto_DataType::TensorProto_DataType_FLOAT16, true, true);
  RunRandomNormalGpuTest(dims1, -1.f, 8.f, 231.f, TensorProto_DataType::TensorProto_DataType_DOUBLE, true, false);
  RunRandomNormalGpuTest(dims1, 0.f, 16.f, 312.f, TensorProto_DataType::TensorProto_DataType_FLOAT16, true, false);
  std::vector<int64_t> dims2{255, 255};
  RunRandomNormalGpuTest(dims2, 1.f, 10.f, 123.f, TensorProto_DataType::TensorProto_DataType_FLOAT, false, false);
  RunRandomNormalGpuTest(dims2, -1.f, 8.f, 231.f, TensorProto_DataType::TensorProto_DataType_DOUBLE, true, true);
  RunRandomNormalGpuTest(dims2, 0.f, 16.f, 312.f, TensorProto_DataType::TensorProto_DataType_FLOAT16, true, false);
}

void RunRandomUniformGpuTest(const std::vector<int64_t> dims, const float low, const float high, const float seed,
                             TensorProto_DataType dtype, bool is_random_like, bool infer_dtype) {
  OpTester test(is_random_like ? "RandomUniformLike" : "RandomUniform");
  test.AddAttribute("low", low);
  test.AddAttribute("high", high);
  test.AddAttribute("seed", seed);
  if (!is_random_like) {
    test.AddAttribute<int64_t>("dtype", dtype);
  } else if (!infer_dtype) {
    // For RandomUniformLike, if not infer dtype, use float as target.
    test.AddAttribute<int64_t>("dtype", TensorProto_DataType::TensorProto_DataType_FLOAT);
  }
  size_t size = 1;
  for (size_t i = 0; i < dims.size(); ++i) {
    size *= static_cast<size_t>(dims[i]);
  }
  if (!is_random_like) {
    test.AddAttribute("shape", dims);
  } else {
    if (dtype == TensorProto_DataType::TensorProto_DataType_FLOAT) {
      std::vector<float> float_data(size, 0.f);
      test.AddInput("X", dims, float_data);
    } else if (dtype == TensorProto_DataType::TensorProto_DataType_DOUBLE) {
      std::vector<double> double_data(size, 0.);
      test.AddInput("X", dims, double_data);
    } else if (dtype == TensorProto_DataType::TensorProto_DataType_FLOAT16) {
      std::vector<float> float_data(size, 0.f);
      std::vector<MLFloat16> fp16_data(size);
      ConvertFloatToMLFloat16(float_data.data(), fp16_data.data(), static_cast<int>(size));
      test.AddInput("X", dims, fp16_data);
    }
  }

  // We'll do our own output verification.
  TensorProto_DataType output_dtype =
      is_random_like && !infer_dtype ? TensorProto_DataType::TensorProto_DataType_FLOAT : dtype;
  if (output_dtype == TensorProto_DataType::TensorProto_DataType_FLOAT) {
    std::vector<float> float_data(size, 0.f);
    test.AddOutput("Y", dims, float_data);
  } else if (output_dtype == TensorProto_DataType::TensorProto_DataType_DOUBLE) {
    std::vector<double> double_data(size, 0.);
    test.AddOutput("Y", dims, double_data);
  } else if (output_dtype == TensorProto_DataType::TensorProto_DataType_FLOAT16) {
    std::vector<float> float_data(size, 0.f);
    std::vector<MLFloat16> fp16_data(size);
    ConvertFloatToMLFloat16(float_data.data(), fp16_data.data(), static_cast<int>(size));
    test.AddOutput("Y", dims, fp16_data);
  }

  auto output_verifier = [&](const std::vector<OrtValue>& fetches, const std::string& provider_type) {
    // Only one output. Each value in output tensoer is between low and high.
    // Mean of output values are near attribute mean of low and high.
    ASSERT_EQ(fetches.size(), 1u);
    const auto& output_tensor = fetches[0].Get<Tensor>();
    if (output_dtype == TensorProto_DataType::TensorProto_DataType_FLOAT) {
      auto output_span = output_tensor.DataAsSpan<float>();
      for (auto value : output_span) {
        ASSERT_GE(value, low);
        ASSERT_LE(value, high);
      }
      float sum = std::accumulate(output_span.begin(), output_span.end(), 0.f);
      ASSERT_NEAR(sum / static_cast<float>(size), (high + low) / 2.f, 0.1f);
    } else if (output_dtype == TensorProto_DataType::TensorProto_DataType_DOUBLE) {
      auto output_span = output_tensor.DataAsSpan<double>();
      for (auto value : output_span) {
        ASSERT_GE(value, static_cast<double>(low));
        ASSERT_LE(value, static_cast<double>(high));
      }
      double sum = std::accumulate(output_span.begin(), output_span.end(), 0.);
      ASSERT_NEAR(sum / static_cast<double>(size), static_cast<double>((high + low) / 2.f), 0.1);
    } else if (output_dtype == TensorProto_DataType::TensorProto_DataType_FLOAT16) {
      auto output_span = output_tensor.DataAsSpan<MLFloat16>();
      float sum = 0.f;
      for (auto value : output_span) {
        float f = value.ToFloat();
        ASSERT_GE(f, low);
        ASSERT_LE(f, high);
        sum += f;
      }
      ASSERT_NEAR(sum / static_cast<float>(size), (high + low) / 2.f, 0.1f);
    }
  };

  test.SetCustomOutputVerifier(output_verifier);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider, kTensorrtExecutionProvider});
}

TEST(Random, RandomUniformGpu) {
  // We will call RandomVectorizedKernel if total_size % 4 == 0, so test two input sizes here.
  std::vector<int64_t> dims1{256, 256};
  RunRandomUniformGpuTest(dims1, 0.f, 10.f, 123.f, TensorProto_DataType::TensorProto_DataType_FLOAT, false, false);
  RunRandomUniformGpuTest(dims1, -10.f, 0.f, 231.f, TensorProto_DataType::TensorProto_DataType_DOUBLE, false, false);
  RunRandomUniformGpuTest(dims1, -5.f, 5.f, 312.f, TensorProto_DataType::TensorProto_DataType_FLOAT16, false, false);
  RunRandomUniformGpuTest(dims1, 0.f, 10.f, 123.f, TensorProto_DataType::TensorProto_DataType_FLOAT, true, true);
  RunRandomUniformGpuTest(dims1, -10.f, 0.f, 231.f, TensorProto_DataType::TensorProto_DataType_DOUBLE, true, true);
  RunRandomUniformGpuTest(dims1, -5.f, 5.f, 312.f, TensorProto_DataType::TensorProto_DataType_FLOAT16, true, true);
  RunRandomUniformGpuTest(dims1, -10.f, 0.f, 231.f, TensorProto_DataType::TensorProto_DataType_DOUBLE, true, false);
  RunRandomUniformGpuTest(dims1, -5.f, 5.f, 312.f, TensorProto_DataType::TensorProto_DataType_FLOAT16, true, false);
  std::vector<int64_t> dims2{255, 255};
  RunRandomUniformGpuTest(dims2, 0.f, 10.f, 123.f, TensorProto_DataType::TensorProto_DataType_FLOAT, false, false);
  RunRandomUniformGpuTest(dims2, -10.f, 0.f, 231.f, TensorProto_DataType::TensorProto_DataType_DOUBLE, true, true);
  RunRandomUniformGpuTest(dims2, -5.f, 5.f, 312.f, TensorProto_DataType::TensorProto_DataType_FLOAT16, true, false);
}
#endif

}  // namespace test
}  // namespace onnxruntime
