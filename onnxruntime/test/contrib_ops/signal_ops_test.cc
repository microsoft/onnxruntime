// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef BUILD_MS_EXPERIMENTAL_OPS

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void TestNaiveDFTFloat(bool is_onesided) {
  OpTester test("DFT", 1, onnxruntime::kMSExperimentalDomain);

  std::vector<int64_t> shape = {1, 5};
  std::vector<int64_t> output_shape = {1, 5, 2};
  output_shape[1] = is_onesided ? (1 + (shape[1] >> 1)) : shape[1];

  std::vector<float> input = {1, 2, 3, 4, 5};
  std::vector<float> expected_output = {
    15.000000f, 0.0000000f,
    -2.499999f, 3.4409550f,
    -2.500000f, 0.8123000f,
    -2.499999f, -0.812299f,
    -2.500003f, -3.440953f
  };

  if (is_onesided) {
    expected_output.resize(6);
  }
  test.AddInput<float>("input", shape, input);
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(is_onesided));
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

static void TestRadix2DFTFloat(bool is_onesided) {
  OpTester test("DFT", 1, onnxruntime::kMSExperimentalDomain);

  std::vector<int64_t> shape = {1, 8};
  std::vector<int64_t> output_shape = {1, 8, 2};
  output_shape[1] = is_onesided ? (1 + (shape[1] >> 1)) : shape[1];

  std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> expected_output = {
    36.000f, 0.000f,
    -4.000f, 9.65685f,
    -4.000f, 4.000f,
    -4.000f, 1.65685f,
    -4.000f, 0.000f,
    -4.000f, -1.65685f,
    -4.000f, -4.000f,
    -4.000f, -9.65685f
  };

  if (is_onesided) {
    expected_output.resize(10);
  }
  test.AddInput<float>("input", shape, input);
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(is_onesided));
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

TEST(MLSignalOpTest, DFTFloat) {
  TestNaiveDFTFloat(false);
  TestNaiveDFTFloat(true);
  TestRadix2DFTFloat(false);
  TestRadix2DFTFloat(true);
}

TEST(MLSignalOpTest, IDFTFloat) {
  OpTester test("IDFT", 1, onnxruntime::kMSExperimentalDomain);
  
  std::vector<int64_t> shape = {1, 5, 2};
  std::vector<float> input =
  {
    15.000000f, 0.0000000f,
    -2.499999f, 3.4409550f,
    -2.500000f, 0.8123000f,
    -2.499999f, -0.812299f,
    -2.500003f, -3.440953f
  };
  std::vector<float> expected_output =
  {
      1.000f, 0.000f,
      2.000f, 0.000f,
      3.000f, 0.000f,
      4.000f, 0.000f,
      5.000f, 0.000f
  };
  
  test.AddInput<float>("input", shape, input);
  test.AddOutput<float>("output", shape, expected_output);
  test.Run();
}

TEST(MLSignalOpTest, STFTFloat) {
  OpTester test("STFT", 1, onnxruntime::kMSExperimentalDomain);

  std::vector<float> signal(64, 1);
  test.AddInput<float>("signal", {1, 64}, signal);
  std::vector<float> window(16, 1);
  test.AddInput<float>("window", {16}, window);
  test.AddInput<int64_t>("frame_length", {}, {16});
  test.AddInput<int64_t>("frame_step", {}, {8});

  std::vector<int64_t> output_shape = {1, 7, 9, 2};
  std::vector<float> expected_output =
  {
    16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
    16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
    16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
    16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
    16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
    16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
    16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f
  };
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

TEST(MLSignalOpTest, HannWindowFloat) {
  OpTester test("HannWindow", 1, onnxruntime::kMSExperimentalDomain);

  std::vector<int64_t> scalar_shape = {};
  std::vector<int64_t> output_shape = {32};
  std::vector<float> expected_output =
  {
    0.000000f, 0.009607f, 0.038060f, 0.084265f, 0.146447f, 0.222215f, 0.308658f, 0.402455f,
    0.500000f, 0.597545f, 0.691342f, 0.777785f, 0.853553f, 0.915735f, 0.961940f, 0.990393f,
    1.000000f, 0.990393f, 0.961940f, 0.915735f, 0.853553f, 0.777785f, 0.691342f, 0.597545f,
    0.500000f, 0.402455f, 0.308658f, 0.222215f, 0.146447f, 0.084265f, 0.038060f, 0.009607f
  };

  test.AddInput<int64_t>("size", scalar_shape, {32});
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

TEST(MLSignalOpTest, HammingWindowFloat) {
  OpTester test("HammingWindow", 1, onnxruntime::kMSExperimentalDomain);
  
  std::vector<int64_t> scalar_shape = {};
  std::vector<int64_t> output_shape = {32};
  std::vector<float> expected_output =
  {
    0.086957f, 0.095728f, 0.121707f, 0.163894f, 0.220669f, 0.289848f, 0.368775f, 0.454415f,
    0.543478f, 0.632541f, 0.718182f, 0.797108f, 0.866288f, 0.923062f, 0.965249f, 0.991228f,
    1.000000f, 0.991228f, 0.965249f, 0.923062f, 0.866288f, 0.797108f, 0.718182f, 0.632541f,
    0.543478f, 0.454415f, 0.368775f, 0.289848f, 0.220669f, 0.163894f, 0.121707f, 0.095728f
  };

  test.AddInput<int64_t>("size", scalar_shape, {32});
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

TEST(MLSignalOpTest, BlackmanWindowFloat) {
  OpTester test("BlackmanWindow", 1, onnxruntime::kMSExperimentalDomain);
  
  std::vector<int64_t> scalar_shape = {};
  std::vector<int64_t> output_shape = {32};
  std::vector<float> expected_output =
  {
    0.000000f, 0.003518f, 0.014629f, 0.034880f, 0.066447f, 0.111600f, 0.172090f, 0.248544f,
    0.340000f, 0.443635f, 0.554773f, 0.667170f, 0.773553f, 0.866350f, 0.938508f, 0.984303f,
    1.000000f, 0.984303f, 0.938508f, 0.866350f, 0.773553f, 0.667170f, 0.554773f, 0.443635f,
    0.340000f, 0.248544f, 0.172090f, 0.111600f, 0.066447f, 0.034880f, 0.014629f, 0.003518f
  };

  test.AddInput<int64_t>("size", scalar_shape, {32});
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

TEST(MLSignalOpTest, MelWeightMatrixFloat) {
  OpTester test("MelWeightMatrix", 1, onnxruntime::kMSExperimentalDomain);

  std::vector<int64_t> scalar_shape = {};
  std::vector<int64_t> output_shape = {9, 8};
  std::vector<float> expected_output =
  {
    1.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 1.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f
  };

  test.AddInput<int64_t>("num_mel_bins", scalar_shape, {8});
  test.AddInput<int64_t>("dft_length", scalar_shape, {16});
  test.AddInput<int64_t>("sample_rate", scalar_shape, {8192});
  test.AddInput<float>("lower_edge_hertz", scalar_shape, {0});
  test.AddInput<float>("upper_edge_hertz", scalar_shape, {8192 / 2.f});
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

#endif