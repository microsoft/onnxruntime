// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/test_random_seed.h"
#include "test/util/include/default_providers.h"

using std::vector;

namespace onnxruntime {
namespace test {

static constexpr int kMinOpsetVersion = 17;
static constexpr int kOpsetVersion20 = 20;

static void TestNaiveDFTFloat(bool onesided, int since_version) {
  OpTester test("DFT", since_version);

  vector<int64_t> shape = {1, 5, 1};
  vector<int64_t> output_shape = {1, 5, 2};
  output_shape[1] = onesided ? (1 + (shape[1] >> 1)) : shape[1];

  vector<float> input = {1, 2, 3, 4, 5};
  vector<float> expected_output = {15.000000f, 0.0000000f, -2.499999f, 3.4409550f, -2.500000f,
                                   0.8123000f, -2.499999f, -0.812299f, -2.500003f, -3.440953f};

  if (onesided) {
    expected_output.resize(6);
  }
  test.AddInput<float>("input", shape, input);
  if (since_version == 20) {
    test.AddInput<int64_t>("dft_length", {}, {5});
    test.AddInput<int64_t>("axis", {}, {-2});
  }
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(onesided));
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

static void TestRadix2DFTFloat(bool onesided, int since_version) {
  OpTester test("DFT", since_version);

  vector<int64_t> shape = {1, 8, 1};
  vector<int64_t> output_shape = {1, 8, 2};
  output_shape[1] = onesided ? (1 + (shape[1] >> 1)) : shape[1];

  vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
  vector<float> expected_output = {36.000f, 0.000f, -4.000f, 9.65685f, -4.000f, 4.000f, -4.000f, 1.65685f,
                                   -4.000f, 0.000f, -4.000f, -1.65685f, -4.000f, -4.000f, -4.000f, -9.65685f};

  if (onesided) {
    expected_output.resize(10);
  }
  test.AddInput<float>("input", shape, input);
  if (since_version == 20) {
    test.AddInput<int64_t>("dft_length", {}, {8});
    test.AddInput<int64_t>("axis", {}, {1});
  }
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(onesided));
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

static void TestInverseFloat(int since_version) {
  OpTester test("DFT", since_version);

  vector<int64_t> shape = {1, 5, 2};
  vector<float> input = {15.000000f, 0.0000000f, -2.499999f, 3.4409550f, -2.500000f,
                         0.8123000f, -2.499999f, -0.812299f, -2.500003f, -3.440953f};
  vector<float> expected_output = {1.000f, 0.000f, 2.000f, 0.000f, 3.000f, 0.000f, 4.000f, 0.000f, 5.000f, 0.000f};

  test.AddInput<float>("input", shape, input);
  if (since_version == 20) {
    test.AddInput<int64_t>("dft_length", {}, {5});
    test.AddInput<int64_t>("axis", {}, {1});
  }
  test.AddAttribute<int64_t>("inverse", static_cast<int64_t>(true));
  test.AddOutput<float>("output", shape, expected_output);
  test.Run();
}

TEST(SignalOpsTest, DFT17_Float_naive) {
  TestNaiveDFTFloat(false, kMinOpsetVersion);
}

TEST(SignalOpsTest, DFT20_Float_naive) {
  TestNaiveDFTFloat(false, kOpsetVersion20);
}

TEST(SignalOpsTest, DFT17_Float_naive_onesided) {
  TestNaiveDFTFloat(true, kMinOpsetVersion);
}

TEST(SignalOpsTest, DFT20_Float_naive_onesided) {
  TestNaiveDFTFloat(true, kOpsetVersion20);
}

TEST(SignalOpsTest, DFT17_Float_radix2) { TestRadix2DFTFloat(false, kMinOpsetVersion); }

TEST(SignalOpsTest, DFT20_Float_radix2) { TestRadix2DFTFloat(false, kOpsetVersion20); }

TEST(SignalOpsTest, DFT17_Float_radix2_onesided) { TestRadix2DFTFloat(true, kMinOpsetVersion); }

TEST(SignalOpsTest, DFT20_Float_radix2_onesided) { TestRadix2DFTFloat(true, kOpsetVersion20); }

TEST(SignalOpsTest, DFT_EmptySignalDimension) {
  for (int opset : {kMinOpsetVersion, kOpsetVersion20}) {
    SCOPED_TRACE(opset);
    OpTester test("DFT", opset);
    test.AddInput<float>("input", {1, 0, 1}, {});
    if (opset == kOpsetVersion20) {
      test.AddOptionalInputEdge<int64_t>();
      test.AddInput<int64_t>("axis", {}, {1});
    } else {
      test.AddAttribute<int64_t>("axis", 1);
    }
    test.AddOutput<float>("output", {1, 0, 2}, {});

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(SignalOpsTest, DFT_EmptySignalWithExplicitLength) {
  OpTester test("DFT", kOpsetVersion20);
  test.AddInput<float>("input", {1, 0, 1}, {});
  test.AddInput<int64_t>("dft_length", {}, {2});
  test.AddInput<int64_t>("axis", {}, {1});
  test.AddOutput<float>("output", {1, 2, 2}, std::vector<float>(4));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "DFT input signal dimension must be greater than zero",
           {}, nullptr, &execution_providers);
}

TEST(SignalOpsTest, DFT20_Float_Bluestein_DftLengthTruncatesInput) {
  OpTester test("DFT", kOpsetVersion20);

  // Non-power-of-2 dft_length selects the Bluestein path.
  // Input length on axis=1 is intentionally larger than dft_length. The trailing input must be ignored.
  vector<int64_t> input_shape = {1, 10, 1};
  vector<float> input = {1.f, 2.f, 3.f, 100.f, 200.f, 300.f, 400.f, 500.f, 600.f, 700.f};
  vector<int64_t> output_shape = {1, 3, 2};
  // clang-format off
  vector<float> expected_output = {
      6.000000f, 0.000000f,
      -1.500000f, 0.866025f,
      -1.500000f, -0.866025f,
  };
  // clang-format on

  test.AddInput<float>("input", input_shape, input);
  test.AddInput<int64_t>("dft_length", {}, {3});
  test.AddInput<int64_t>("axis", {}, {1});
  test.AddOutput<float>("output", output_shape, expected_output);
  test.SetOutputAbsErr("output", 0.001f);
  // DML path is deprecated for this test scenario.
  test.ConfigExcludeEps({kDmlExecutionProvider});
  test.RunWithConfig();
}

TEST(SignalOpsTest, DFT17_Float_inverse) {
  TestInverseFloat(kMinOpsetVersion);
}

TEST(SignalOpsTest, DFT20_Float_inverse) {
  TestInverseFloat(kOpsetVersion20);
}

// Tests that FFT(FFT(x), inverse=true) == x
static void TestDFTInvertible(bool complex, int since_version) {
  // TODO: test dft_length
  class DFTInvertibleTester : public OpTester {
   public:
    DFTInvertibleTester(int64_t axis, int since_version) : OpTester("DFT", since_version), axis_(axis) {}

   protected:
    void AddNodes(Graph& graph, vector<NodeArg*>& graph_inputs, vector<NodeArg*>& graph_outputs,
                  vector<std::function<void(Node& node)>>& add_attribute_funcs) override {
      // Create an intermediate output
      vector<NodeArg*> intermediate_outputs{&graph.GetOrCreateNodeArg("dft_output", graph_outputs[0]->TypeAsProto())};

      // call base implementation to add the DFT node.
      OpTester::AddNodes(graph, graph_inputs, intermediate_outputs, add_attribute_funcs);
      if (this->Opset() < kOpsetVersion20) {
        OpTester::AddAttribute("axis", axis_);
      } else {
        assert(intermediate_outputs.size() == 1);
        assert(graph_inputs.size() == 3);
        intermediate_outputs.push_back(graph_inputs[1]);
        intermediate_outputs.push_back(graph_inputs[2]);
      }

      Node& inverse = graph.AddNode("inverse", "DFT", "inverse", intermediate_outputs, graph_outputs);
      inverse.AddAttribute("inverse", static_cast<int64_t>(true));
      if (this->Opset() < kOpsetVersion20) {
        inverse.AddAttribute("axis", axis_);
      }
    }

   private:
    int64_t axis_;
  };

  RandomValueGenerator random(GetTestRandomSeed());
  // TODO(smk2007): Add tests for different dft_length values.
  constexpr int64_t num_batches = 2;
  for (int64_t axis = 0; axis < 2; axis += 1) {
    for (int64_t signal_dim1 = 2; signal_dim1 <= 5; signal_dim1 += 1) {
      for (int64_t signal_dim2 = 2; signal_dim2 <= 5; signal_dim2 += 1) {
        if (axis == 0 && since_version < kOpsetVersion20)
          continue;
        DFTInvertibleTester test(axis, since_version);
        vector<int64_t> input_shape{num_batches, signal_dim1, signal_dim2, 1 + (complex ? 1 : 0)};
        vector<float> input_data = random.Uniform<float>(input_shape, -100.f, 100.f);
        test.AddInput("input", input_shape, input_data);

        if (since_version >= kOpsetVersion20) {
          test.AddInput<int64_t>("", {0}, {});
          test.AddInput<int64_t>("axis", {1}, {axis});
        }

        vector<int64_t> output_shape(input_shape);
        vector<float>* output_data_p;
        vector<float> output_data;
        if (complex) {
          output_data_p = &input_data;
        } else {  // real -> (real, imaginary) with imaginary == 0.
          output_shape[3] = 2;
          output_data.resize(input_data.size() * 2, 0);
          for (size_t i = 0; i < input_data.size(); i += 1) {
            output_data[i * 2] = input_data[i];
          }
          output_data_p = &output_data;
        }
        test.AddOutput<float>("output", output_shape, *output_data_p);
        test.SetOutputAbsErr("output", 0.0002f);
        test.Run();
      }
    }
  }
}

TEST(SignalOpsTest, DFT17_invertible_real) {
  TestDFTInvertible(false, kMinOpsetVersion);
}

TEST(SignalOpsTest, DFT20_invertible_real) {
  TestDFTInvertible(false, kOpsetVersion20);
}

TEST(SignalOpsTest, DFT17_invertible_complex) {
  TestDFTInvertible(true, kMinOpsetVersion);
}

TEST(SignalOpsTest, DFT20_invertible_complex) {
  TestDFTInvertible(true, kOpsetVersion20);
}

TEST(SignalOpsTest, STFTFloat) {
  OpTester test("STFT", kMinOpsetVersion);

  vector<float> signal(64, 1);
  test.AddInput<float>("signal", {1, 64, 1}, signal);
  test.AddInput<int64_t>("frame_step", {}, {8});
  vector<float> window(16, 1);
  test.AddInput<float>("window", {16}, window);
  test.AddInput<int64_t>("frame_length", {}, {16});

  vector<int64_t> output_shape = {1, 7, 9, 2};
  vector<float> expected_output = {
      16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
      0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
      0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
      16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
      0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
      0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
      16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
      0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
      0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
      16.000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f,
      0.0000f, 0.000f, 0.0000f, 0.000f, 0.0000f, 0.000f};
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

static void TestSTFTInvalidFrameStep(int64_t frame_step) {
  OpTester test("STFT", kMinOpsetVersion);

  vector<float> signal(64, 1);
  test.AddInput<float>("signal", {1, 64, 1}, signal);
  test.AddInput<int64_t>("frame_step", {}, {frame_step});
  vector<float> window(16, 1);
  test.AddInput<float>("window", {16}, window);
  test.AddInput<int64_t>("frame_length", {}, {16});

  vector<int64_t> output_shape = {1, 7, 9, 2};
  vector<float> expected_output(1 * 7 * 9 * 2, 0.f);
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Config(OpTester::ExpectResult::kExpectFailure, "frame_step must be greater than zero");
  test.ConfigExcludeEps({kDmlExecutionProvider});
  test.RunWithConfig();
}

TEST(SignalOpsTest, STFTFrameStepMustBePositive) {
  TestSTFTInvalidFrameStep(0);
  TestSTFTInvalidFrameStep(-1);
}

template <typename T>
static void TestSTFTComplexInputBatched() {
  OpTester test("STFT", kMinOpsetVersion);
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(false));

  constexpr int64_t batch_size = 2;
  constexpr int64_t signal_size = 128;
  constexpr int64_t signal_components = 2;
  constexpr int64_t frame_length = 32;
  constexpr int64_t frame_step = 16;
  constexpr int64_t n_dfts = 7;
  constexpr int64_t dft_output_size = frame_length;
  constexpr int64_t output_components = 2;

  vector<T> signal(batch_size * signal_size * signal_components, static_cast<T>(0));
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    const T signal_value = batch_idx == 0 ? static_cast<T>(1) : static_cast<T>(99);
    for (int64_t sample_idx = 0; sample_idx < signal_size; ++sample_idx) {
      signal[(batch_idx * signal_size + sample_idx) * signal_components] = signal_value;
    }
  }

  test.AddInput<T>("signal", {batch_size, signal_size, signal_components}, signal);
  test.AddInput<int64_t>("frame_step", {}, {frame_step});
  test.AddOptionalInputEdge<T>();
  test.AddInput<int64_t>("frame_length", {}, {frame_length});

  vector<int64_t> output_shape = {batch_size, n_dfts, dft_output_size, output_components};
  vector<T> expected_output(batch_size * n_dfts * dft_output_size * output_components, static_cast<T>(0));
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    const T dc_value = (batch_idx == 0 ? static_cast<T>(1) : static_cast<T>(99)) * static_cast<T>(frame_length);
    for (int64_t frame_idx = 0; frame_idx < n_dfts; ++frame_idx) {
      expected_output[((batch_idx * n_dfts + frame_idx) * dft_output_size) * output_components] = dc_value;
    }
  }

  test.AddOutput<T>("output", output_shape, expected_output);
  test.SetOutputAbsErr("output", 0.001f);
  // DML does not consistently match these CPU STFT validation/regression paths in Windows GPU CI.
  test.ConfigExcludeEps({kDmlExecutionProvider});
  test.RunWithConfig();
}

TEST(SignalOpsTest, STFTFloatComplexInputBatched) {
  TestSTFTComplexInputBatched<float>();
}

TEST(SignalOpsTest, STFTDoubleComplexInputBatched) {
  TestSTFTComplexInputBatched<double>();
}

template <typename T>
static void TestSTFTComplexInputWithWindow(bool use_radix2) {
  OpTester test("STFT", kMinOpsetVersion);
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(false));

  const int64_t frame_length = use_radix2 ? 4 : 3;
  vector<T> signal(static_cast<size_t>(frame_length) * 2, static_cast<T>(1));
  test.AddInput<T>("signal", {1, frame_length, 2}, signal);
  test.AddInput<int64_t>("frame_step", {}, {frame_length});

  vector<T> window(static_cast<size_t>(frame_length));
  for (int64_t i = 0; i < frame_length; ++i) {
    window[static_cast<size_t>(i)] = static_cast<T>(i + 1);
  }
  test.AddInput<T>("window", {frame_length}, window);
  test.AddInput<int64_t>("frame_length", {}, {frame_length});

  const vector<T> expected_output = use_radix2
                                        ? vector<T>{10, 10, -4, 0, -2, -2, 0, -4}
                                        : vector<T>{6, 6, static_cast<T>(-2.3660254), static_cast<T>(-0.6339746),
                                                    static_cast<T>(-0.6339746), static_cast<T>(-2.3660254)};
  test.AddOutput<T>("output", {1, 1, frame_length, 2}, expected_output);
  test.SetOutputAbsErr("output", 0.0001f);
  test.ConfigExcludeEps({kDmlExecutionProvider});
  test.RunWithConfig();
}

TEST(SignalOpsTest, STFTFloatComplexInputWithWindowRadix2) {
  TestSTFTComplexInputWithWindow<float>(true);
}

TEST(SignalOpsTest, STFTDoubleComplexInputWithWindowRadix2) {
  TestSTFTComplexInputWithWindow<double>(true);
}

TEST(SignalOpsTest, STFTFloatComplexInputWithWindowBluestein) {
  TestSTFTComplexInputWithWindow<float>(false);
}

TEST(SignalOpsTest, STFTDoubleComplexInputWithWindowBluestein) {
  TestSTFTComplexInputWithWindow<double>(false);
}

TEST(SignalOpsTest, HannWindowFloat) {
  OpTester test("HannWindow", kMinOpsetVersion);

  vector<int64_t> scalar_shape = {};
  vector<int64_t> output_shape = {32};
  vector<float> expected_output = {0.000000f, 0.009607f, 0.038060f, 0.084265f, 0.146447f, 0.222215f, 0.308658f,
                                   0.402455f, 0.500000f, 0.597545f, 0.691342f, 0.777785f, 0.853553f, 0.915735f,
                                   0.961940f, 0.990393f, 1.000000f, 0.990393f, 0.961940f, 0.915735f, 0.853553f,
                                   0.777785f, 0.691342f, 0.597545f, 0.500000f, 0.402455f, 0.308658f, 0.222215f,
                                   0.146447f, 0.084265f, 0.038060f, 0.009607f};

  test.AddInput<int64_t>("size", scalar_shape, {32});
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

TEST(SignalOpsTest, HammingWindowFloat) {
  OpTester test("HammingWindow", kMinOpsetVersion);

  vector<int64_t> scalar_shape = {};
  vector<int64_t> output_shape = {32};
  vector<float> expected_output =  //
      {0.086957f, 0.095728f, 0.121707f, 0.163894f, 0.220669f, 0.289848f, 0.368775f, 0.454415f,
       0.543478f, 0.632541f, 0.718182f, 0.797108f, 0.866288f, 0.923062f, 0.965249f, 0.991228f,
       1.000000f, 0.991228f, 0.965249f, 0.923062f, 0.866288f, 0.797108f, 0.718182f, 0.632541f,
       0.543478f, 0.454415f, 0.368775f, 0.289848f, 0.220669f, 0.163894f, 0.121707f, 0.095728f};

  test.AddInput<int64_t>("size", scalar_shape, {32});
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

TEST(SignalOpsTest, BlackmanWindowFloat) {
  OpTester test("BlackmanWindow", kMinOpsetVersion);

  vector<int64_t> scalar_shape = {};
  vector<int64_t> output_shape = {32};
  vector<float> expected_output =  //
      {0.000000f, 0.003518f, 0.014629f, 0.034880f, 0.066447f, 0.111600f, 0.172090f, 0.248544f,
       0.340000f, 0.443635f, 0.554773f, 0.667170f, 0.773553f, 0.866350f, 0.938508f, 0.984303f,
       1.000000f, 0.984303f, 0.938508f, 0.866350f, 0.773553f, 0.667170f, 0.554773f, 0.443635f,
       0.340000f, 0.248544f, 0.172090f, 0.111600f, 0.066447f, 0.034880f, 0.014629f, 0.003518f};

  test.AddInput<int64_t>("size", scalar_shape, {32});
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

TEST(SignalOpsTest, MelWeightMatrixFloat) {
  OpTester test("MelWeightMatrix", kMinOpsetVersion);

  vector<int64_t> scalar_shape = {};
  vector<int64_t> output_shape = {9, 8};
  vector<float> expected_output = {
      1.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
      0.000000f, 1.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
      0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
      0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
      0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
      0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
      0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,
      0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f};

  test.AddInput<int64_t>("num_mel_bins", scalar_shape, {8});
  test.AddInput<int64_t>("dft_length", scalar_shape, {16});
  test.AddInput<int64_t>("sample_rate", scalar_shape, {8192});
  test.AddInput<float>("lower_edge_hertz", scalar_shape, {0});
  test.AddInput<float>("upper_edge_hertz", scalar_shape, {8192 / 2.f});
  test.AddOutput<float>("output", output_shape, expected_output);
  test.Run();
}

// IRFFT tests - inverse one-sided DFT (complex to real)
static void TestIRFFTRadix2Float(int since_version) {
  OpTester test("DFT", since_version);

  // One-sided complex input (result of RFFT on 8 real samples)
  vector<int64_t> input_shape = {1, 5, 2};  // floor(8/2) + 1 = 5 frequency bins
  vector<float> input = {36.000f, 0.000f, -4.000f, 9.65685f, -4.000f, 4.000f,
                         -4.000f, 1.65685f, -4.000f, 0.000f};

  // Expected real output (should match original signal from RFFT)
  vector<int64_t> output_shape = {1, 8, 1};
  vector<float> expected_output = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  test.AddInput<float>("input", input_shape, input);
  if (since_version == 20) {
    test.AddInput<int64_t>("dft_length", {}, {8});
    test.AddInput<int64_t>("axis", {}, {1});
  }
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(true));
  test.AddAttribute<int64_t>("inverse", static_cast<int64_t>(true));
  test.AddOutput<float>("output", output_shape, expected_output);
  test.SetOutputAbsErr("output", 0.0001f);
  test.ConfigExcludeEps({kDmlExecutionProvider});
  test.RunWithConfig();
}

static void TestIRFFTNaiveFloat(int since_version) {
  OpTester test("DFT", since_version);

  // One-sided complex input (result of RFFT on 5 real samples)
  vector<int64_t> input_shape = {1, 3, 2};  // floor(5/2) + 1 = 3 frequency bins
  vector<float> input = {15.000000f, 0.0000000f, -2.499999f, 3.4409550f, -2.500000f, 0.8123000f};

  // Expected real output
  vector<int64_t> output_shape = {1, 5, 1};
  vector<float> expected_output = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  test.AddInput<float>("input", input_shape, input);
  // dft_length is required for IRFFT to distinguish between even and odd lengths
  test.AddInput<int64_t>("dft_length", {}, {5});
  if (since_version == 20) {
    test.AddInput<int64_t>("axis", {}, {-2});
  }
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(true));
  test.AddAttribute<int64_t>("inverse", static_cast<int64_t>(true));
  test.AddOutput<float>("output", output_shape, expected_output);
  test.SetOutputAbsErr("output", 0.0001f);
  test.ConfigExcludeEps({kDmlExecutionProvider});
  test.RunWithConfig();
}

// Test RFFT -> IRFFT round trip
static void TestRFFTIRFFTRoundTrip(int since_version) {
  class RFFTIRFFTTester : public OpTester {
   public:
    explicit RFFTIRFFTTester(int since_version) : OpTester("DFT", since_version) {}

   protected:
    void AddNodes(Graph& graph, vector<NodeArg*>& graph_inputs, vector<NodeArg*>& graph_outputs,
                  vector<std::function<void(Node& node)>>& add_attribute_funcs) override {
      // Create intermediate output for RFFT
      ONNX_NAMESPACE::TypeProto intermediate_type;
      intermediate_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      vector<NodeArg*> intermediate_outputs;
      intermediate_outputs.push_back(&graph.GetOrCreateNodeArg("rfft_output", &intermediate_type));

      // Add RFFT node (forward one-sided)
      OpTester::AddNodes(graph, graph_inputs, intermediate_outputs, add_attribute_funcs);

      if (this->Opset() < kOpsetVersion20) {
        // For opset 17-19, just pass through
      } else {
        // For opset 20, pass dft_length and axis to IRFFT
        assert(graph_inputs.size() == 3);
        intermediate_outputs.push_back(graph_inputs[1]);
        intermediate_outputs.push_back(graph_inputs[2]);
      }

      // Add IRFFT node (inverse one-sided)
      Node& irfft = graph.AddNode("irfft", "DFT", "inverse one-sided", intermediate_outputs, graph_outputs);
      irfft.AddAttribute("onesided", static_cast<int64_t>(true));
      irfft.AddAttribute("inverse", static_cast<int64_t>(true));
    }
  };

  RFFTIRFFTTester test(since_version);

  // Real input signal
  vector<int64_t> input_shape = {2, 8, 1};
  vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                              8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  test.AddInput<float>("input", input_shape, input_data);
  if (since_version >= kOpsetVersion20) {
    test.AddInput<int64_t>("dft_length", {}, {8});
    test.AddInput<int64_t>("axis", {}, {1});
  }
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(true));

  // Output should match input (round trip)
  test.AddOutput<float>("output", input_shape, input_data);
  test.SetOutputAbsErr("output", 0.001f);
  test.ConfigExcludeEps({kDmlExecutionProvider});
  test.RunWithConfig();
}

TEST(SignalOpsTest, DFT17_IRFFT_radix2) {
  TestIRFFTRadix2Float(kMinOpsetVersion);
}

TEST(SignalOpsTest, DFT20_IRFFT_radix2) {
  TestIRFFTRadix2Float(kOpsetVersion20);
}

TEST(SignalOpsTest, DFT17_IRFFT_naive) {
  TestIRFFTNaiveFloat(kMinOpsetVersion);
}

TEST(SignalOpsTest, DFT20_IRFFT_naive) {
  TestIRFFTNaiveFloat(kOpsetVersion20);
}

TEST(SignalOpsTest, DFT17_RFFT_IRFFT_roundtrip) {
  TestRFFTIRFFTRoundTrip(kMinOpsetVersion);
}

TEST(SignalOpsTest, DFT20_RFFT_IRFFT_roundtrip) {
  TestRFFTIRFFTRoundTrip(kOpsetVersion20);
}

// Length 7 is prime, exercising the non-radix code paths (Bluestein on CPU, direct DFT on GPU EPs).
static void TestPrimeLengthDFTFloat(bool onesided, int since_version) {
  OpTester test("DFT", since_version);

  vector<int64_t> shape = {1, 7, 1};
  vector<float> input = {1, 2, 3, 4, 5, 6, 7};

  vector<int64_t> output_shape = {1, onesided ? 4 : 7, 2};
  vector<float> expected_output = {28.00000f, 0.00000f, -3.50000f, 7.26782f,
                                   -3.50000f, 2.79116f, -3.50000f, 0.79885f};
  if (!onesided) {
    expected_output.insert(expected_output.end(),
                           {-3.50000f, -0.79885f, -3.50000f, -2.79116f, -3.50000f, -7.26782f});
  }

  test.AddInput<float>("input", shape, input);
  if (since_version == 20) {
    test.AddInput<int64_t>("dft_length", {}, {7});
    test.AddInput<int64_t>("axis", {}, {1});
  }
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(onesided));
  test.AddOutput<float>("output", output_shape, expected_output);
  test.SetOutputAbsErr("output", 0.0002f);
  test.ConfigExcludeEps({kDmlExecutionProvider});
  test.RunWithConfig();
}

TEST(SignalOpsTest, DFT17_Float_prime_length) { TestPrimeLengthDFTFloat(false, kMinOpsetVersion); }

TEST(SignalOpsTest, DFT20_Float_prime_length) { TestPrimeLengthDFTFloat(false, kOpsetVersion20); }

TEST(SignalOpsTest, DFT17_Float_prime_length_onesided) { TestPrimeLengthDFTFloat(true, kMinOpsetVersion); }

TEST(SignalOpsTest, DFT20_Float_prime_length_onesided) { TestPrimeLengthDFTFloat(true, kOpsetVersion20); }

// dft_length longer than the signal zero-pads before the transform; padding to 8 stays on radix
// paths while padding to 7 lands on the non-radix fallback.
static void TestZeroPaddedDFTFloat(int64_t dft_length, int since_version) {
  OpTester test("DFT", since_version);

  vector<int64_t> shape = {1, 5, 1};
  vector<float> input = {1, 2, 3, 4, 5};

  const vector<float> expected_padded_to_8 = {
      15.00000f, 0.00000f, -5.41421f, -7.24264f, 3.00000f, 2.00000f, -2.58579f, -1.24264f,
      3.00000f, 0.00000f, -2.58579f, 1.24264f, 3.00000f, -2.00000f, -5.41421f, 7.24264f};
  const vector<float> expected_padded_to_7 = {
      15.00000f, 0.00000f, -6.52930f, -4.05456f, 3.46346f, -1.43004f, -0.93416f, 2.45265f,
      -0.93416f, -2.45265f, 3.46346f, 1.43004f, -6.52930f, 4.05456f};
  const vector<float>& expected_output = dft_length == 8 ? expected_padded_to_8 : expected_padded_to_7;

  test.AddInput<float>("input", shape, input);
  test.AddInput<int64_t>("dft_length", {}, {dft_length});
  if (since_version == 20) {
    test.AddInput<int64_t>("axis", {}, {1});
  }
  test.AddOutput<float>("output", {1, dft_length, 2}, expected_output);
  test.SetOutputAbsErr("output", 0.0002f);
  test.ConfigExcludeEps({kDmlExecutionProvider});
  test.RunWithConfig();
}

TEST(SignalOpsTest, DFT17_Float_zero_padded_radix2) { TestZeroPaddedDFTFloat(8, kMinOpsetVersion); }

TEST(SignalOpsTest, DFT20_Float_zero_padded_radix2) { TestZeroPaddedDFTFloat(8, kOpsetVersion20); }

TEST(SignalOpsTest, DFT17_Float_zero_padded_prime) { TestZeroPaddedDFTFloat(7, kMinOpsetVersion); }

TEST(SignalOpsTest, DFT20_Float_zero_padded_prime) { TestZeroPaddedDFTFloat(7, kOpsetVersion20); }

// Runs only against EPs that register DFT for float16 (the CPU kernel is float/double only).
// Kernels accumulate in float32 internally, so the tolerance covers float16 output quantization.
static void TestRadix2DFTFloat16(int since_version) {
  OpTester test("DFT", since_version);

  vector<int64_t> shape = {1, 8, 1};
  vector<int64_t> output_shape = {1, 8, 2};

  vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
  vector<float> expected_output = {36.000f, 0.000f, -4.000f, 9.65685f, -4.000f, 4.000f, -4.000f, 1.65685f,
                                   -4.000f, 0.000f, -4.000f, -1.65685f, -4.000f, -4.000f, -4.000f, -9.65685f};

  test.AddInput<MLFloat16>("input", shape, ToFloat16(input));
  if (since_version == 20) {
    test.AddInput<int64_t>("dft_length", {}, {8});
    test.AddInput<int64_t>("axis", {}, {1});
  }
  test.AddOutput<MLFloat16>("output", output_shape, ToFloat16(expected_output));
  test.SetOutputAbsErr("output", 0.05f);
  test.ConfigExcludeEps({kDmlExecutionProvider});
  test.RunWithConfig();
}

TEST(SignalOpsTest, DFT17_Float16_radix2) { TestRadix2DFTFloat16(kMinOpsetVersion); }

TEST(SignalOpsTest, DFT20_Float16_radix2) { TestRadix2DFTFloat16(kOpsetVersion20); }

// dft_length can imply a different half-spectrum bin count (floor(dft_length/2) + 1) than the
// IRFFT input provides; missing bins are treated as zero and extra bins are ignored. The CPU
// kernel implements these semantics on its Bluestein path but not yet on its radix-2 path, so
// radix-2 length cases exclude the CPU EP until that is fixed.
static void TestIRFFTSpectrumLengthMismatchFloat(const vector<float>& input, int64_t input_bins,
                                                 int64_t dft_length, const vector<float>& expected_output,
                                                 int since_version, bool exclude_cpu) {
  if (exclude_cpu && nullptr == DefaultWebGpuExecutionProvider().get()) {
    GTEST_SKIP() << "Test requires the WebGPU execution provider when the CPU radix-2 path is excluded.";
  }

  OpTester test("DFT", since_version);

  test.AddInput<float>("input", {1, input_bins, 2}, input);
  test.AddInput<int64_t>("dft_length", {}, {dft_length});
  if (since_version == 20) {
    test.AddInput<int64_t>("axis", {}, {1});
  }
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(true));
  test.AddAttribute<int64_t>("inverse", static_cast<int64_t>(true));
  test.AddOutput<float>("output", {1, dft_length, 1}, expected_output);
  test.SetOutputAbsErr("output", 0.001f);
  std::unordered_set<std::string> excluded_providers{kDmlExecutionProvider};
  if (exclude_cpu) {
    excluded_providers.insert(kCpuExecutionProvider);
  }
  test.ConfigExcludeEps(excluded_providers);
  test.RunWithConfig();
}

TEST(SignalOpsTest, DFT17_IRFFT_underprovided_spectrum_radix2) {
  TestIRFFTSpectrumLengthMismatchFloat(
      {1, 0, 2, 1, 3, -2}, 3, 8,
      {1.375000f, 0.801777f, -0.875000f, -0.905330f, 0.375000f, 0.448223f, -0.375000f, 0.155330f},
      kMinOpsetVersion, /*exclude_cpu=*/true);
}

TEST(SignalOpsTest, DFT20_IRFFT_underprovided_spectrum_radix2) {
  TestIRFFTSpectrumLengthMismatchFloat(
      {1, 0, 2, 1, 3, -2}, 3, 8,
      {1.375000f, 0.801777f, -0.875000f, -0.905330f, 0.375000f, 0.448223f, -0.375000f, 0.155330f},
      kOpsetVersion20, /*exclude_cpu=*/true);
}

TEST(SignalOpsTest, DFT20_IRFFT_underprovided_spectrum_prime) {
  TestIRFFTSpectrumLengthMismatchFloat(
      {1, 0, 2, 1, 3, -2}, 3, 7,
      {1.571429f, 0.642126f, -1.283041f, -0.408290f, 0.733165f, -0.230072f, -0.025316f},
      kOpsetVersion20, /*exclude_cpu=*/false);
}

TEST(SignalOpsTest, DFT20_IRFFT_overprovided_spectrum_radix2) {
  TestIRFFTSpectrumLengthMismatchFloat(
      {1, 0, 2, 1, 3, -2, 4, 0.5f, -1, 0, 5, 5, 6, -6}, 7, 8,
      {2.250000f, 0.131282f, -0.875000f, -0.161612f, -0.750000f, 1.368718f, -0.625000f, -0.338388f},
      kOpsetVersion20, /*exclude_cpu=*/true);
}

// Test 2D complex input (single 1D signal without batch dimension)
static void TestDFT2DComplex(int since_version) {
  OpTester test("DFT", since_version);

  // 2D complex input: [signal_length, 2]
  // This represents a single 1D complex signal without a batch dimension
  vector<int64_t> input_shape = {8, 2};
  vector<float> input = {
      1.0f, 0.0f,  // complex(1, 0)
      2.0f, 0.0f,  // complex(2, 0)
      3.0f, 0.0f,  // complex(3, 0)
      4.0f, 0.0f,  // complex(4, 0)
      5.0f, 0.0f,  // complex(5, 0)
      6.0f, 0.0f,  // complex(6, 0)
      7.0f, 0.0f,  // complex(7, 0)
      8.0f, 0.0f   // complex(8, 0)
  };

  // Expected output: DFT of the complex input
  // Should have same shape [8, 2] for complex output
  vector<int64_t> output_shape = {8, 2};
  vector<float> expected_output = {
      36.000f, 0.000f,     // bin 0
      -4.000f, 9.65685f,   // bin 1
      -4.000f, 4.000f,     // bin 2
      -4.000f, 1.65685f,   // bin 3
      -4.000f, 0.000f,     // bin 4
      -4.000f, -1.65685f,  // bin 5
      -4.000f, -4.000f,    // bin 6
      -4.000f, -9.65685f   // bin 7
  };

  test.AddInput<float>("input", input_shape, input);
  if (since_version == 20) {
    test.AddInput<int64_t>("dft_length", {}, {8});
    test.AddInput<int64_t>("axis", {}, {0});  // axis=0 for 2D input
  } else {
    // For Opset 17, set axis attribute explicitly
    test.AddAttribute<int64_t>("axis", static_cast<int64_t>(0));
  }
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(false));
  test.AddOutput<float>("output", output_shape, expected_output);
  test.SetOutputAbsErr("output", 0.0001f);
  test.Run();
}

TEST(SignalOpsTest, DFT17_2D_complex) {
  TestDFT2DComplex(kMinOpsetVersion);
}

TEST(SignalOpsTest, DFT20_2D_complex) {
  TestDFT2DComplex(kOpsetVersion20);
}

// Test 2D real input with onesided=true (forward transform - RFFT)
static void TestDFT2DComplexOnesided(int since_version) {
  OpTester test("DFT", since_version);

  // 2D real input: [signal_length, 1]
  // This represents a single 1D real signal without a batch dimension
  vector<int64_t> input_shape = {8, 1};
  vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  // Expected output: RFFT of the real input (one-sided)
  // Output shape [5, 2] contains only positive frequency bins (8/2 + 1 = 5)
  vector<int64_t> output_shape = {5, 2};
  vector<float> expected_output = {
      36.000f, 0.000f,    // bin 0 (DC)
      -4.000f, 9.65685f,  // bin 1
      -4.000f, 4.000f,    // bin 2
      -4.000f, 1.65685f,  // bin 3
      -4.000f, 0.000f     // bin 4 (Nyquist for even length)
  };

  test.AddInput<float>("input", input_shape, input);
  if (since_version == 20) {
    test.AddInput<int64_t>("dft_length", {}, {8});
    test.AddInput<int64_t>("axis", {}, {0});  // axis=0 for 2D input
  } else {
    // For Opset 17, set axis attribute explicitly
    test.AddAttribute<int64_t>("axis", static_cast<int64_t>(0));
  }
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(true));
  test.AddOutput<float>("output", output_shape, expected_output);
  test.SetOutputAbsErr("output", 0.0001f);
  test.Run();
}

TEST(SignalOpsTest, DFT17_2D_complex_onesided) {
  TestDFT2DComplexOnesided(kMinOpsetVersion);
}

TEST(SignalOpsTest, DFT20_2D_complex_onesided) {
  TestDFT2DComplexOnesided(kOpsetVersion20);
}

// Test 2D complex input with onesided=true and inverse=true (IRFFT)
static void TestDFT2DComplexOnesidedInverse(int since_version) {
  OpTester test("DFT", since_version);

  // 2D complex input: [onesided_length, 2]
  // This represents a onesided complex spectrum (5 bins for original signal length 8)
  vector<int64_t> input_shape = {5, 2};
  vector<float> input = {
      36.000f, 0.000f,    // bin 0 (DC)
      -4.000f, 9.65685f,  // bin 1
      -4.000f, 4.000f,    // bin 2
      -4.000f, 1.65685f,  // bin 3
      -4.000f, 0.000f     // bin 4 (Nyquist)
  };

  // Expected output: IRFFT reconstructs the real signal [8, 1]
  vector<int64_t> output_shape = {8, 1};
  vector<float> expected_output = {
      1.0f,  // Should reconstruct original real values
      2.0f,
      3.0f,
      4.0f,
      5.0f,
      6.0f,
      7.0f,
      8.0f};

  test.AddInput<float>("input", input_shape, input);
  if (since_version == 20) {
    test.AddInput<int64_t>("dft_length", {}, {8});
    test.AddInput<int64_t>("axis", {}, {0});  // axis=0 for 2D input
  } else {
    // For Opset 17, set axis attribute explicitly
    test.AddAttribute<int64_t>("axis", static_cast<int64_t>(0));
  }
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(true));
  test.AddAttribute<int64_t>("inverse", static_cast<int64_t>(true));
  test.AddOutput<float>("output", output_shape, expected_output);
  test.SetOutputAbsErr("output", 0.0001f);
  test.ConfigExcludeEps({kDmlExecutionProvider});
  test.RunWithConfig();
}

TEST(SignalOpsTest, DFT17_2D_complex_onesided_inverse) {
  TestDFT2DComplexOnesidedInverse(kMinOpsetVersion);
}

TEST(SignalOpsTest, DFT20_2D_complex_onesided_inverse) {
  TestDFT2DComplexOnesidedInverse(kOpsetVersion20);
}

// Test 2D real input (single 1D signal without batch dimension)
static void TestDFT2DReal(int since_version) {
  OpTester test("DFT", since_version);

  // 2D real input: [signal_length, 1]
  // This represents a single 1D real signal without a batch dimension
  vector<int64_t> input_shape = {8, 1};
  vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  // Expected output: RFFT of the real input (one-sided)
  vector<int64_t> output_shape = {5, 2};  // floor(8/2) + 1 = 5 bins
  vector<float> expected_output = {
      36.000f, 0.000f,
      -4.000f, 9.65685f,
      -4.000f, 4.000f,
      -4.000f, 1.65685f,
      -4.000f, 0.000f};

  test.AddInput<float>("input", input_shape, input);
  if (since_version == 20) {
    test.AddInput<int64_t>("dft_length", {}, {8});
    test.AddInput<int64_t>("axis", {}, {0});  // axis=0 for 2D input
  } else {
    // For Opset 17, set axis attribute explicitly
    test.AddAttribute<int64_t>("axis", static_cast<int64_t>(0));
  }
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(true));
  test.AddOutput<float>("output", output_shape, expected_output);
  test.SetOutputAbsErr("output", 0.0001f);
  test.Run();
}

TEST(SignalOpsTest, DFT17_2D_real) {
  TestDFT2DReal(kMinOpsetVersion);
}

TEST(SignalOpsTest, DFT20_2D_real) {
  TestDFT2DReal(kOpsetVersion20);
}

// ---- WebGPU FFT engine coverage -------------------------------------------------------------------
// The WebGPU kernels pick among single-pass Stockham, real-packed, four-step and Bluestein FFTs by
// length; these cases compare every path against a double-precision reference DFT.

namespace {

vector<float> RandomValues(size_t count, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  vector<float> values(count);
  for (auto& v : values) v = dist(rng);
  return values;
}

constexpr double kPi = 3.141592653589793238462643383279502884;

// In-place radix-2 FFT of a power-of-two-sized sequence with exp(sign * 2*pi*i*j*k / n) twiddles, unscaled.
void Radix2Fft(vector<std::complex<double>>& a, double sign) {
  const size_t n = a.size();
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (size_t len = 2; len <= n; len <<= 1) {
    for (size_t k = 0; k < len / 2; ++k) {
      const double angle = sign * 2.0 * kPi * static_cast<double>(k) / static_cast<double>(len);
      const std::complex<double> w{std::cos(angle), std::sin(angle)};
      for (size_t i = 0; i < n; i += len) {
        const auto u = a[i + k];
        const auto v = a[i + k + len / 2] * w;
        a[i + k] = u + v;
        a[i + k + len / 2] = u - v;
      }
    }
  }
}

// Length-`length` DFT of `signal`, with exp(sign * 2*pi*i*j*k / length) twiddles, in double precision.
// Short lengths use the direct sum; longer ones a radix-2 FFT, through Bluestein's chirp-z identity
// jk = (j^2 + k^2 - (k - j)^2) / 2 when the length is not a power of two, so large cases stay cheap.
vector<std::complex<double>> ReferenceTransform(const vector<std::complex<double>>& signal, int64_t length,
                                                bool inverse) {
  const double sign = inverse ? 1.0 : -1.0;
  const size_t n = static_cast<size_t>(length);
  vector<std::complex<double>> spectrum(n);
  if (length <= 512) {
    vector<std::complex<double>> twiddles(n);
    for (size_t i = 0; i < n; ++i) {
      const double angle = sign * 2.0 * kPi * static_cast<double>(i) / static_cast<double>(length);
      twiddles[i] = {std::cos(angle), std::sin(angle)};
    }
    for (size_t k = 0; k < n; ++k) {
      std::complex<double> acc = 0;
      size_t index = 0;
      for (size_t j = 0; j < n; ++j) {
        acc += signal[j] * twiddles[index];
        index += k;
        if (index >= n) index -= n;
      }
      spectrum[k] = acc;
    }
  } else if ((n & (n - 1)) == 0) {
    spectrum.assign(signal.begin(), signal.begin() + length);
    Radix2Fft(spectrum, sign);
  } else {
    // chirp[j] = exp(sign * i*pi * j^2 / n), with j^2 reduced mod 2n to keep the angle exact.
    vector<std::complex<double>> chirp(n);
    for (size_t j = 0; j < n; ++j) {
      const double angle = sign * kPi * static_cast<double>((j * j) % (2 * n)) / static_cast<double>(n);
      chirp[j] = {std::cos(angle), std::sin(angle)};
    }
    size_t m = 1;
    while (m < 2 * n - 1) m <<= 1;
    vector<std::complex<double>> a(m), b(m);
    for (size_t j = 0; j < n; ++j) {
      a[j] = signal[j] * chirp[j];
      b[j] = std::conj(chirp[j]);
      if (j > 0) b[m - j] = b[j];
    }
    Radix2Fft(a, -1.0);
    Radix2Fft(b, -1.0);
    for (size_t i = 0; i < m; ++i) a[i] *= b[i];
    Radix2Fft(a, 1.0);
    for (size_t k = 0; k < n; ++k) spectrum[k] = chirp[k] * a[k] / static_cast<double>(m);
  }
  if (inverse) {
    for (auto& v : spectrum) v /= static_cast<double>(length);
  }
  return spectrum;
}

float FftTolerance(const vector<float>& expected, int64_t length) {
  float max_abs = 0.0f;
  for (float v : expected) max_abs = std::max(max_abs, std::abs(v));
  return 5e-5f * std::log2(2.0f * static_cast<float>(length)) * max_abs + 1e-5f;
}

void RunOnWebGpu(OpTester& test) {
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.ConfigEps(std::move(execution_providers)).RunWithConfig();
}

// DFT along axis 1 of an [outer, signal_length, inner, components] tensor.
void TestWebGpuDft(int64_t outer, int64_t signal_length, int64_t inner, int64_t components, int64_t length,
                   bool inverse, bool onesided) {
  if (nullptr == DefaultWebGpuExecutionProvider().get()) {
    GTEST_SKIP() << "Test requires the WebGPU execution provider.";
  }
  SCOPED_TRACE(::testing::Message() << "outer=" << outer << " signal_length=" << signal_length << " inner=" << inner
                                    << " components=" << components << " length=" << length
                                    << " inverse=" << inverse << " onesided=" << onesided);
  const bool hermitian = inverse && onesided;
  const int64_t output_length = onesided && !inverse ? length / 2 + 1 : length;
  const int64_t output_components = hermitian ? 1 : 2;
  const vector<float> input = RandomValues(static_cast<size_t>(outer * signal_length * inner * components),
                                           static_cast<uint32_t>(length * 31 + signal_length));
  vector<float> expected(static_cast<size_t>(outer * output_length * inner * output_components));

  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t w = 0; w < inner; ++w) {
      auto sample = [&](int64_t j) {
        const size_t off = static_cast<size_t>(((o * signal_length + j) * inner + w) * components);
        return std::complex<double>(input[off], components == 2 ? input[off + 1] : 0.0f);
      };
      vector<std::complex<double>> signal(static_cast<size_t>(length));
      for (int64_t j = 0; j < length; ++j) {
        if (hermitian) {
          const int64_t half = length / 2 + 1;
          const int64_t provided = std::min(signal_length, half);
          if (j < half) {
            signal[static_cast<size_t>(j)] = j < provided ? sample(j) : 0.0;
          } else if (length - j < provided) {
            signal[static_cast<size_t>(j)] = std::conj(sample(length - j));
          }
        } else if (j < signal_length) {
          signal[static_cast<size_t>(j)] = sample(j);
        }
      }
      const auto spectrum = ReferenceTransform(signal, length, inverse);
      for (int64_t k = 0; k < output_length; ++k) {
        const size_t off = static_cast<size_t>(((o * output_length + k) * inner + w) * output_components);
        expected[off] = static_cast<float>(spectrum[static_cast<size_t>(k)].real());
        if (output_components == 2) expected[off + 1] = static_cast<float>(spectrum[static_cast<size_t>(k)].imag());
      }
    }
  }

  OpTester test("DFT", kOpsetVersion20);
  test.AddInput<float>("input", {outer, signal_length, inner, components}, input);
  test.AddInput<int64_t>("dft_length", {}, {length});
  test.AddInput<int64_t>("axis", {}, {1});
  test.AddAttribute<int64_t>("inverse", static_cast<int64_t>(inverse));
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(onesided));
  test.AddOutput<float>("output", {outer, output_length, inner, output_components}, expected);
  test.SetOutputAbsErr("output", FftTolerance(expected, length));
  RunOnWebGpu(test);
}

void TestWebGpuDftAllModes(int64_t length, int64_t outer) {
  TestWebGpuDft(outer, length, 1, 2, length, false, false);        // complex forward
  TestWebGpuDft(outer, length, 1, 2, length, true, false);         // complex inverse
  TestWebGpuDft(outer, length, 1, 1, length, false, false);        // real forward, full spectrum
  TestWebGpuDft(outer, length, 1, 1, length, false, true);         // RFFT
  TestWebGpuDft(outer, length / 2 + 1, 1, 2, length, true, true);  // IRFFT
}

}  // namespace

// 13-smooth lengths that fit in workgroup memory: single-pass Stockham (radices 2..13), including
// short transforms packed several per workgroup.
TEST(SignalOpsTest, WebGpuDFT_SinglePass) {
  for (int64_t length : {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 30, 49, 64, 100, 120, 121, 169, 256, 384, 1000,
                         1024, 2048}) {
    TestWebGpuDftAllModes(length, 5);
  }
}

// Prime and other non-13-smooth lengths: Bluestein chirp-z.
TEST(SignalOpsTest, WebGpuDFT_Bluestein) {
  for (int64_t length : {17, 31, 34, 97, 257, 1009, 2053}) {
    TestWebGpuDftAllModes(length, 3);
  }
}

// Lengths beyond workgroup memory: four-step FFT (and four-step inside Bluestein).
TEST(SignalOpsTest, WebGpuDFT_FourStep) {
  for (int64_t length : {4096, 6000, 8192, 16384, 4099}) {
    TestWebGpuDftAllModes(length, 2);
  }
}

// Transforms along a non-trailing signal axis, so consecutive transforms interleave in memory.
TEST(SignalOpsTest, WebGpuDFT_StridedRows) {
  for (int64_t length : {16, 60, 17, 5000}) {
    TestWebGpuDft(2, length, 3, 2, length, false, false);
    TestWebGpuDft(2, length, 3, 1, length, false, true);
    TestWebGpuDft(2, length / 2 + 1, 3, 2, length, true, true);
  }
}

// dft_length that zero-pads or truncates the signal.
TEST(SignalOpsTest, WebGpuDFT_PadAndTruncate) {
  for (auto [signal_length, length] : std::vector<std::pair<int64_t, int64_t>>{{5, 8}, {12, 8}, {100, 257}, {300, 17}, {3000, 6000}}) {
    TestWebGpuDft(3, signal_length, 1, 2, length, false, false);
    TestWebGpuDft(3, signal_length, 1, 1, length, false, true);
    TestWebGpuDft(3, signal_length, 1, 2, length, true, true);
  }
}

namespace {

void TestWebGpuStft(int64_t batch, int64_t signal_length, int64_t components, int64_t frame_length,
                    int64_t frame_step, bool use_window, bool pass_frame_length, bool onesided, bool rank2) {
  if (nullptr == DefaultWebGpuExecutionProvider().get()) {
    GTEST_SKIP() << "Test requires the WebGPU execution provider.";
  }
  SCOPED_TRACE(::testing::Message() << "batch=" << batch << " signal_length=" << signal_length
                                    << " components=" << components << " frame_length=" << frame_length
                                    << " frame_step=" << frame_step << " window=" << use_window
                                    << " onesided=" << onesided);
  const vector<float> signal = RandomValues(static_cast<size_t>(batch * signal_length * components),
                                            static_cast<uint32_t>(frame_length * 7 + frame_step));
  const vector<float> window = RandomValues(static_cast<size_t>(frame_length), 1234u);
  const int64_t frames = (signal_length - frame_length) / frame_step + 1;
  const int64_t bins = onesided ? frame_length / 2 + 1 : frame_length;

  vector<float> expected(static_cast<size_t>(batch * frames * bins * 2));
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t f = 0; f < frames; ++f) {
      vector<std::complex<double>> frame(static_cast<size_t>(frame_length));
      for (int64_t j = 0; j < frame_length; ++j) {
        const size_t off = static_cast<size_t>((b * signal_length + f * frame_step + j) * components);
        const double scale = use_window ? window[static_cast<size_t>(j)] : 1.0;
        frame[static_cast<size_t>(j)] = std::complex<double>(signal[off], components == 2 ? signal[off + 1] : 0.0f) *
                                        scale;
      }
      const auto spectrum = ReferenceTransform(frame, frame_length, false);
      for (int64_t k = 0; k < bins; ++k) {
        const size_t off = static_cast<size_t>(((b * frames + f) * bins + k) * 2);
        expected[off] = static_cast<float>(spectrum[static_cast<size_t>(k)].real());
        expected[off + 1] = static_cast<float>(spectrum[static_cast<size_t>(k)].imag());
      }
    }
  }

  OpTester test("STFT", kMinOpsetVersion);
  test.AddAttribute<int64_t>("onesided", static_cast<int64_t>(onesided));
  if (rank2) {
    test.AddInput<float>("signal", {batch, signal_length}, signal);
  } else {
    test.AddInput<float>("signal", {batch, signal_length, components}, signal);
  }
  test.AddInput<int64_t>("frame_step", {}, {frame_step});
  if (use_window) {
    test.AddInput<float>("window", {frame_length}, window);
  } else {
    test.AddOptionalInputEdge<float>();
  }
  if (pass_frame_length) {
    test.AddInput<int64_t>("frame_length", {}, {frame_length});
  }
  test.AddOutput<float>("output", {batch, frames, bins, 2}, expected);
  test.SetOutputAbsErr("output", FftTolerance(expected, frame_length));
  RunOnWebGpu(test);
}

}  // namespace

TEST(SignalOpsTest, WebGpuSTFT) {
  // Real signal, onesided (real-packed path), with the usual audio frame sizes.
  TestWebGpuStft(2, 4000, 1, 400, 160, true, true, true, false);
  TestWebGpuStft(2, 4000, 1, 512, 128, true, false, true, false);
  TestWebGpuStft(1, 20000, 1, 2048, 512, true, true, true, true);
  // Odd, prime (Bluestein) and four-step frame lengths.
  TestWebGpuStft(2, 3000, 1, 255, 100, true, true, true, false);
  TestWebGpuStft(2, 3000, 1, 251, 64, true, true, true, false);
  TestWebGpuStft(1, 20000, 1, 8192, 4000, true, true, true, false);
  // Full spectrum, complex signals, no window.
  TestWebGpuStft(3, 1000, 1, 64, 32, true, true, false, false);
  TestWebGpuStft(3, 1000, 2, 64, 30, true, true, false, false);
  TestWebGpuStft(3, 1000, 2, 48, 30, false, true, true, false);
  TestWebGpuStft(2, 777, 2, 97, 41, false, true, false, false);
}

}  // namespace test
}  // namespace onnxruntime
