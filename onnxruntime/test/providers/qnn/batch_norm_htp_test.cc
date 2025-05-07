// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/framework/float16.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Computes the mean and variance of inputs within a channel.
// Requires an input with rank >= 3
template <typename FLOAT_TYPE>
static void ComputeChannelMeanAndVar(const std::vector<FLOAT_TYPE>& input_data, const std::vector<int64_t>& input_shape,
                                     std::vector<FLOAT_TYPE>& mean_vals, std::vector<FLOAT_TYPE>& var_vals) {
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
    mean_vals[i] = FLOAT_TYPE{};
    var_vals[i] = FLOAT_TYPE{};
  }

  // Compute running sum of elements within each channel. The running sum is stored in the mean_vals array directly.
  for (size_t b = 0; b < num_batches; b++) {
    const size_t batch_start = b * batch_stride;

    for (size_t c = 0; c < num_channels; c++) {
      const size_t chan_start = batch_start + (c * channel_stride);

      for (size_t i = chan_start; i < chan_start + channel_stride; i++) {
        mean_vals[c] = FLOAT_TYPE(mean_vals[c] + input_data[i]);
      }
    }
  }

  // Divide sums by the number of elements in a channel to get the mean.
  for (size_t c = 0; c < num_channels; c++) {
    mean_vals[c] = FLOAT_TYPE(mean_vals[c] / FLOAT_TYPE(static_cast<float>(num_batches * channel_stride)));
  }

  // Compute running sum of deviations from mean within each channel. The running sum is stored in the var_vals array directly.
  for (size_t b = 0; b < num_batches; b++) {
    const size_t batch_start = b * batch_stride;

    for (size_t c = 0; c < num_channels; c++) {
      const size_t chan_start = batch_start + (c * channel_stride);

      for (size_t i = chan_start; i < chan_start + channel_stride; i++) {
        const FLOAT_TYPE deviation = FLOAT_TYPE(input_data[i] - mean_vals[c]);
        var_vals[c] = FLOAT_TYPE(var_vals[c] + (deviation * deviation));
      }
    }
  }

  // Divide sums by the number of elements in a channel to get the variance.
  for (size_t c = 0; c < num_channels; c++) {
    var_vals[c] = FLOAT_TYPE(var_vals[c] / FLOAT_TYPE(static_cast<float>(num_batches * channel_stride)));
  }
}

template <typename FLOAT_TYPE>
static GetTestModelFn BuildBatchNormTestCase(const TestInputDef<FLOAT_TYPE>& input_def,
                                             const TestInputDef<FLOAT_TYPE>& scale_def,
                                             const TestInputDef<FLOAT_TYPE>& bias_def) {
  ORT_ENFORCE(input_def.IsRawData());  // Need raw data to compute mean and variance inputs.

  return [input_def, scale_def, bias_def](ModelTestBuilder& builder) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];

    NodeArg* input = MakeTestInput<FLOAT_TYPE>(builder, input_def);
    NodeArg* scale = MakeTestInput<FLOAT_TYPE>(builder, scale_def);
    NodeArg* bias = MakeTestInput<FLOAT_TYPE>(builder, bias_def);

    std::vector<FLOAT_TYPE> mean_vals(num_channels);
    std::vector<FLOAT_TYPE> var_vals(num_channels);
    ComputeChannelMeanAndVar<FLOAT_TYPE>(input_data, input_shape, mean_vals, var_vals);

    NodeArg* mean = builder.MakeInitializer<FLOAT_TYPE>({num_channels}, mean_vals);
    NodeArg* var = builder.MakeInitializer<FLOAT_TYPE>({num_channels}, var_vals);
    NodeArg* output = builder.MakeOutput();
    builder.AddNode("BatchNormalization", {input, scale, bias, mean, var}, {output});
  };
}

template <typename InputQType, typename ScaleQType>
GetTestQDQModelFn<InputQType> BuildQDQBatchNormTestCase(const TestInputDef<float>& input_def,
                                                        const TestInputDef<float>& scale_def,
                                                        const TestInputDef<float>& bias_def) {
  ORT_ENFORCE(input_def.IsRawData());  // Need raw data to compute mean and variance inputs.

  return [input_def, scale_def, bias_def](ModelTestBuilder& builder,
                                          std::vector<QuantParams<InputQType>>& output_qparams) {
    const auto& input_shape = input_def.GetShape();
    const auto& input_data = input_def.GetRawData();
    const int64_t num_channels = input_shape[1];
    bool symmetric = sizeof(InputQType) == sizeof(uint16_t);
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<InputQType> input_qparams = GetTestInputQuantParams<InputQType>(input_def, symmetric);
    NodeArg* input_qdq = AddQDQNodePair<InputQType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    NodeArg* scale = MakeTestInput(builder, scale_def);
    QuantParams<ScaleQType> scale_qparams = GetTestInputQuantParams<ScaleQType>(scale_def);
    NodeArg* scale_qdq = AddQDQNodePair<ScaleQType>(builder, scale, scale_qparams.scale, scale_qparams.zero_point);

    NodeArg* bias_qdq;
    // bias (as int32) => DQ =>
    bias_qdq = MakeTestQDQBiasInput(builder, bias_def, input_qparams.scale * scale_qparams.scale, true);

    std::vector<float> mean_vals(num_channels);
    std::vector<float> var_vals(num_channels);
    ComputeChannelMeanAndVar(input_data, input_shape, mean_vals, var_vals);

    NodeArg* mean = builder.MakeInitializer<float>({num_channels}, mean_vals);
    NodeArg* var = builder.MakeInitializer<float>({num_channels}, var_vals);

    auto* batchnorm_output = builder.MakeIntermediate();
    builder.AddNode("BatchNormalization", {input_qdq, scale_qdq, bias_qdq, mean, var},
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
template <typename InputQType, typename ScaleQType>
static void RunBatchNormQDQTest(const TestInputDef<float>& input_def,
                                const TestInputDef<float>& scale_def,
                                const TestInputDef<float>& bias_def,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  TestQDQModelAccuracy(BuildBatchNormTestCase(input_def, scale_def, bias_def),
                       BuildQDQBatchNormTestCase<InputQType, ScaleQType>(input_def, scale_def, bias_def),
                       provider_options,
                       21,
                       expected_ep_assignment,
                       tolerance);
}

static void RunBatchNormFP16Test(const TestInputDef<float>& input_def,
                                 const TestInputDef<float>& scale_def,
                                 const TestInputDef<float>& bias_def,
                                 ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestInputDef<MLFloat16> input_fp16_def = ConvertToFP16InputDef(input_def);
  TestInputDef<MLFloat16> scale_fp16_def = ConvertToFP16InputDef(scale_def);
  TestInputDef<MLFloat16> bias_fp16_def = ConvertToFP16InputDef(bias_def);

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  TestFp16ModelAccuracy(BuildBatchNormTestCase<float>(input_def, scale_def, bias_def),
                        BuildBatchNormTestCase<MLFloat16>(input_fp16_def, scale_fp16_def, bias_fp16_def),
                        provider_options,
                        11,
                        expected_ep_assignment);
}

// BatchNor QDQ model, input with rank 2.
TEST_F(QnnHTPBackendTests, BatchNormRank2) {
  constexpr int64_t num_channels = 2;

  RunBatchNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({4, num_channels}, false,
                                                            {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f}),  // Input data
                                        TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),                    // Scale initializer
                                        TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),                    // Bias initializer
                                        ExpectedEPNodeAssignment::All);
}

// TODO: FIX TRANSLATION!!!
// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 3.
// Accuracy issue with Linux simulator, not sure with Android device
// Inaccuracy detected for output 'output_0', element 1
// output_range=4.8666362762451172, tolerance=0.40000000596046448%.
// Expected val (f32@CPU_EP): 1.0999999046325684
// qdq@QNN_EP val: -0.17176364362239838 (err: 1.2717635631561279, err/output_range: 26.132291793823242%)
// qdq@CPU_EP val: 1.1069211959838867 (err: 0.0069212913513183594, err/output_range: 0.14221921563148499%)
// abs(qdq@QNN_EP - qdq@CPU_EP) / output_range = 25.990072250366211%
//
// Inaccuracy detected for output 'output_0', element 2
// output_range=4.8666362762451172, tolerance=0.40000000596046448%.
// Expected val (f32@CPU_EP): 2.3247356414794922
// qdq@QNN_EP val: -0.17176364362239838 (err: 2.4964993000030518, err/output_range: 51.298248291015625%)
// qdq@CPU_EP val: 2.3474364280700684 (err: 0.022700786590576172, err/output_range: 0.46645742654800415%)
#if defined(_WIN32)
TEST_F(QnnHTPBackendTests, BatchNorm1D) {
  constexpr int64_t num_channels = 2;

  RunBatchNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, num_channels, 3}, false,
                                                            {-5.0f, -4.0f, -3.0f, 0.0f, 2.0f, 5.0f}),  // Input data
                                        TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),       // Scale initializer
                                        TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),       // Bias initializer
                                        ExpectedEPNodeAssignment::All);
}
#endif

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, BatchNorm2D_a8w8) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
                                        TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
                                        TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
                                        ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, BatchNorm2D_a16w8) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormQDQTest<uint16_t, uint8_t>(TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
                                         TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
                                         TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
                                         ExpectedEPNodeAssignment::All);
}

// Test FP16 BatchNormalization on the HTP backend.
TEST_F(QnnHTPBackendTests, BatchNorm_FP16) {
  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  RunBatchNormFP16Test(TestInputDef<float>({2, num_channels, 2, 2}, false, input_data),  // Input data
                       TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),          // Scale initializer
                       TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),          // Bias initializer
                       ExpectedEPNodeAssignment::All);
}

// Test FP32 BatchNormalization on the HTP backend with the enable_htp_fp16_precision option enabled
// to run it with fp16 precision.
TEST_F(QnnHTPBackendTests, BatchNorm_FP32_as_FP16) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = "htp";
  provider_options["enable_htp_fp16_precision"] = "1";

  constexpr int64_t num_channels = 2;
  std::vector<float> input_data = {-8.0f, -6.0f, -4.0f, -2.0f, 0.0f, 1.1f, 3.3f, 8.0f,
                                   -7.0f, -5.0f, -3.0f, -1.0f, 0.0f, 2.1f, 4.3f, 7.0f};

  auto input_def = TestInputDef<float>({2, num_channels, 2, 2}, false, input_data);
  auto scale_def = TestInputDef<float>({num_channels}, true, {1.0f, 2.0f});
  auto bias_def = TestInputDef<float>({num_channels}, true, {1.1f, 2.1f});
  auto model_fn = BuildBatchNormTestCase<float>(input_def, scale_def, bias_def);

  RunQnnModelTest(model_fn,
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All,
                  0.01f);  // abs err
}

// Check that QNN compiles DQ -> BatchNormalization -> Q as a single unit.
// Use an input of rank 5. QNN BatchNormalization doesn't support 5D on HTP
TEST_F(QnnHTPBackendTests, BatchNorm3D) {
  constexpr int64_t num_channels = 2;
  constexpr int64_t num_elems = 1 * num_channels * 3 * 4 * 5;
  RunBatchNormQDQTest<uint8_t, uint8_t>(TestInputDef<float>({1, num_channels, 3, 4, 5}, false,
                                                            std::vector<float>(num_elems)),       // Input data (all zeros)
                                        TestInputDef<float>({num_channels}, true, {1.0f, 2.0f}),  // Scale initializer
                                        TestInputDef<float>({num_channels}, true, {1.1f, 2.1f}),  // Bias initializer
                                        ExpectedEPNodeAssignment::None);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
