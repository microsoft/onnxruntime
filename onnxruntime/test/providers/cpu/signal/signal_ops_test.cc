// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
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
  for (int64_t axis = 1; axis < 2; axis += 1) {
    for (int64_t signal_dim1 = 2; signal_dim1 <= 5; signal_dim1 += 1) {
      for (int64_t signal_dim2 = 2; signal_dim2 <= 5; signal_dim2 += 1) {
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

}  // namespace test
}  // namespace onnxruntime
