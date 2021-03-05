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

  float scale = 10.f;
  float mean = 0.f;
  float seed = 123.f;

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
  test.Run();
}

void RunRandomNormalLike3DFloat(bool infer_dtype = false) {
  OpTester test("RandomNormalLike");

  std::vector<int64_t> dims{2, 2, 3};

  float scale = 10.f;
  float mean = 0.f;
  float seed = 123.f;

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

  test.Run();
}

TEST(Random, RandomNormalLike3DDouble) {
  RunRandomNormalLike3DFloat();
}

TEST(Random, RandomNormalLikeInferDType) {
  const bool infer_dtype = true;
  RunRandomNormalLike3DFloat(infer_dtype);
}

TEST(Random, RandomUniform1DFloat) {
  OpTester test("RandomUniform");

  std::vector<int64_t> dims{10};

  float low = 0.f;
  float high = 100.f;
  float seed = 123.f;

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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

void RunRandomUniformLikeTest(bool infer_dtype = false) {
  OpTester test("RandomUniformLike");

  std::vector<int64_t> dims{2, 6};

  float low = 0.f;
  float high = 100.f;
  float seed = 123.f;

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

  test.Run();
}

TEST(Random, RandomUniformLike2DDouble) {
  RunRandomUniformLikeTest();
}

TEST(Random, RandomUniformLikeInferDType) {
  const bool infer_dtype = true;
  RunRandomUniformLikeTest(infer_dtype);
}

TEST(Random, InvalidDType) {
  float seed = 123.f;

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

    float low = 0.f;
    float high = 100.f;

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

    float scale = 10.f;
    float mean = 0.f;

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

    float low = 0.f;
    float high = 100.f;

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

  const int64_t num_samples = 10;
  const float seed = 1618.f;
  const int batch_size = 2;
  const int num_classes = 3;

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
#elif defined(__MACH__) || defined(__ANDROID__) || defined(__FreeBSD__) || defined(ENABLE_ORT_WASM)
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
    const int64_t num_samples = 10;
    const int batch_size = 2;
    const float seed = 1618.f;

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
#elif defined(__MACH__) || defined(__ANDROID__) || defined(__FreeBSD__) || defined(ENABLE_ORT_WASM)
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

  const int64_t num_samples = 10;
  const int batch_size = 2;
  const int num_classes = 3;
  const float seed = 1618.f;

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
}  // namespace test
}  // namespace onnxruntime
