// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Computes the mean and variance of inputs within a channel.
// Requires an input with rank >= 3
static void ComputeChannelMeanAndVar(const std::vector<float>& input_data, const std::vector<int64_t>& input_shape,
                                     std::vector<float>& mean_vals, std::vector<float>& var_vals) {
  const size_t input_rank = input_shape.size();
  const size_t num_batches = input_shape[0];
  const size_t num_channels = input_shape[1];

  size_t batch_stride = 1;
  for (size_t i = 1; i < input_rank; i++) {
    batch_stride *= input_shape[i];
  }
  const size_t channel_stride = batch_stride / num_channels;

  assert(mean_vals.size() == num_channels);
  assert(var_vals.size() == num_channels);
  for (size_t i = 0; i < num_channels; i++) {
    mean_vals[i] = 0.0f;
    var_vals[i] = 0.0f;
  }

  // Compute running sum of elements within each channel. The running sum is stored in the mean_vals array directly.
  for (size_t b = 0; b < num_batches; b++) {
    const size_t batch_start = b * batch_stride;

    for (size_t c = 0; c < num_channels; c++) {
      const size_t chan_start = batch_start + (c * channel_stride);

      for (size_t i = chan_start; i < chan_start + channel_stride; i++) {
        mean_vals[c] += input_data[i];
      }
    }
  }

  // Divide sums by the number of elements in a channel to get the mean.
  for (size_t c = 0; c < num_channels; c++) {
    mean_vals[c] /= static_cast<float>(num_batches * channel_stride);
  }

  // Compute running sum of deviations from mean within each channel. The running sum is stored in the var_vals array directly.
  for (size_t b = 0; b < num_batches; b++) {
    const size_t batch_start = b * batch_stride;

    for (size_t c = 0; c < num_channels; c++) {
      const size_t chan_start = batch_start + (c * channel_stride);

      for (size_t i = chan_start; i < chan_start + channel_stride; i++) {
        const float deviation = input_data[i] - mean_vals[c];
        var_vals[c] += (deviation * deviation);
      }
    }
  }

  // Divide sums by the number of elements in a channel to get the variance.
  for (size_t c = 0; c < num_channels; c++) {
    var_vals[c] /= static_cast<float>(num_batches * channel_stride);
  }
}

static GetTestModelFn BuildBatchNormTestCase(const TestInputDef<float>& input_def,
                                             const TestInputDef<float>& scale_def,
                                             const TestInputDef<float>& bias_def) {
  ORT_ENFORCE(input_def.IsRawData());            // Need raw data to compute mean and variance inputs.
  ORT_ENFORCE(input_def.GetShape().size() > 2);  // Need at least rank 3 data for convenience.

  return [input_def, scale_def, bias_def](ModelTestBuilder& builder) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];

    NodeArg* input = MakeTestInput(builder, input_def);
    NodeArg* scale = MakeTestInput(builder, scale_def);
    NodeArg* bias = MakeTestInput(builder, bias_def);

    std::vector<float> mean_vals(num_channels);
    std::vector<float> var_vals(num_channels);
    ComputeChannelMeanAndVar(input_data, input_shape, mean_vals, var_vals);

    NodeArg* mean = builder.MakeInitializer<float>({num_channels}, mean_vals);
    NodeArg* var = builder.MakeInitializer<float>({num_channels}, var_vals);
    NodeArg* output = builder.MakeOutput();
    builder.AddNode("BatchNormalization", {input, scale, bias, mean, var}, {output});
  };
}

template <typename InputQType, typename ScaleQType, typename BiasQType>
GetTestQDQModelFn<InputQType> BuildQDQBatchNormTestCase(const TestInputDef<float>& input_def,
                                                        const TestInputDef<float>& scale_def,
                                                        const TestInputDef<float>& bias_def) {
  ORT_ENFORCE(input_def.IsRawData());            // Need raw data to compute mean and variance inputs.
  ORT_ENFORCE(input_def.GetShape().size() > 2);  // Need at least rank 3 data for convenience.

  return [input_def, scale_def, bias_def](ModelTestBuilder& builder,
                                          std::vector<QuantParams<InputQType>>& output_qparams) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];

    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<InputQType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    NodeArg* scale = MakeTestInput(builder, scale_def);
    QuantParams<ScaleQType> scale_qparams = GetTestInputQuantParams<ScaleQType>(scale_def);
    NodeArg* scale_qdq = AddQDQNodePair<ScaleQType>(builder, scale, scale_qparams.scale, scale_qparams.zero_point);

    NodeArg* bias = MakeTestInput(builder, bias_def);
    QuantParams<BiasQType> bias_qparams = GetTestInputQuantParams<BiasQType>(bias_def);
    NodeArg* bias_qdq = AddQDQNodePair<BiasQType>(builder, bias, bias_qparams.scale, bias_qparams.zero_point);

    std::vector<float> mean_vals(num_channels);
    std::vector<float> var_vals(num_channels);
    ComputeChannelMeanAndVar(input_data, input_shape, mean_vals, var_vals);

    NodeArg* mean = builder.MakeInitializer<float>({num_channels}, mean_vals);
    QuantParams<InputQType> mean_qparams = GetDataQuantParams(mean_vals);
    NodeArg* mean_qdq = AddQDQNodePair<InputQType>(builder, mean, mean_qparams.scale, mean_qparams.zero_point);

    NodeArg* var = builder.MakeInitializer<float>({num_channels}, var_vals);
    QuantParams<InputQType> var_qparams = GetDataQuantParams(var_vals);
    NodeArg* var_qdq = AddQDQNodePair<InputQType>(builder, var, var_qparams.scale, var_qparams.zero_point);

    auto* batchnorm_output = builder.MakeIntermediate();
    builder.AddNode("BatchNormalization", {input_qdq, scale_qdq, bias_qdq, mean_qdq, var_qdq},
                    {batchnorm_output});

    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, batchnorm_output, output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

/**
 * Runs an BatchNormalization model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param input_shape The input's shape.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 */
static void RunBatchNormQDQTest(const TestInputDef<float>& input_def,
                                const TestInputDef<float>& scale_def,
                                const TestInputDef<float>& bias_def,
                                ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  TestQDQModelAccuracy(BuildBatchNormTestCase(input_def, scale_def, bias_def),
                       BuildQDQBatchNormTestCase<uint8_t, uint8_t, uint8_t>(input_def, scale_def, bias_def),
                       provider_options,
                       11,
                       expected_ep_assignment,
                       1e-5f);
}

// TODO: FIX TRANSLATION!!!
// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 3.
// QNN v2.13
// Inaccuracy detected for output 'output', element 4.
// Output quant params: scale=0.019084848463535309, zero_point=9.
// Expected val: 1.7755576372146606
// QNN QDQ val: 2.9963212013244629 (err 1.2207635641098022)
// CPU QDQ val: 0.82064849138259888 (err 0.95490914583206177)
TEST_F(QnnHTPBackendTests, DISABLED_BatchNorm1D) {
  constexpr int64_t num_channels = 2;

  RunBatchNormQDQTest(TestInputDef<float>({1, num_channels, 3}, false, {-5.0f, -4.0f, -3.0f, 0.0f, 2.0f, 5.0f}),  // Input data
                      TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),                                    // Scale initializer
                      TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),                                    // Bias initializer
                      ExpectedEPNodeAssignment::All);
}

// TODO: FIX TRANSLATION!!!
// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 4.
// QNN v2.13
// Inaccuracy detected for output 'output', element 14.
// Output quant params: scale=0.023071292787790298, zero_point=19.
// Expected val: 2.8554618358612061
// QNN QDQ val: 5.3294687271118164 (err 2.4740068912506104)
// CPU QDQ val: 1.6611330509185791 (err 1.194328784942627)
TEST_F(QnnHTPBackendTests, DISABLED_BatchNorm2D) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormQDQTest(TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
                      TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
                      TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
                      ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 5. QNN BatchNormalization doesn't support 5D on HTP
TEST_F(QnnHTPBackendTests, BatchNorm3D) {
  constexpr int64_t num_channels = 2;
  constexpr int64_t num_elems = 1 * num_channels * 3 * 4 * 5;
  RunBatchNormQDQTest(TestInputDef<float>({1, num_channels, 3, 4, 5}, false, std::vector<float>(num_elems)),  // Input data (all zeros)
                      TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),                                // Scale initializer
                      TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),                                // Bias initializer
                      ExpectedEPNodeAssignment::None);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif