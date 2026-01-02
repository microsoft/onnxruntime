// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <filesystem>
#include <variant>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
/**
 * Tests the accuracy of using batch multiplier on QNN EP by running 3 inferences:
 *
 * 1. Run data with "batch multiplier batch size" on CPU EP (with model compiled with batch multiplier batch size) - baseline
 * 2. Run data with "batch multiplier batch size" on QNN HTP (with model compiled with batch multiplier batch size)
 * 3. Run data with "batch multiplier batch size" on QNN HTP (with model compiled with original batch size)
 *
 * This function checks that running #3 is at least as accurate (+- small tolerance) as running #2.
 * We primarily measure accuracy by comparing both #2 and #3 to the baseline (#1).
 *
 * \param bm_model_fn Function that builds the model with "batch multiplier batch size".
 * \param ori_model_fn Function that builds the model with "original batch size".
 * \param qnn_options QNN EP provider options.
 * \param opset_version The opset version.
 * \param expected_ep_assignment Describes which nodes should be assigned to the EP.
 * \param tolerance The percent tolerance (as fraction) QNN HTP using batch multiplier results are allowed to differ from without batch multiplier
 *                  on QNN HTP. This tolerance is a percentage of the output range.
 * \param log_severity The logger's severity setting.
 * \param qnn_ctx_model_path Optional path to a QNN context cache model.
 */
inline void TestModelBatchMultiplierAccuracy(
    const GetTestModelFn& bm_model_fn,
    const GetTestModelFn& ori_model_fn,
    const ProviderOptions& qnn_options,
    int opset_version,
    ExpectedEPNodeAssignment expected_ep_assignment,
    float tolerance = 0.004f,
    logging::Severity log_severity = logging::Severity::kERROR,
    const std::string& qnn_ctx_model_path = "",
    const std::unordered_map<std::string, std::string>& session_option_pairs = {}) {
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(log_severity);

  // 1. Create model with batch multiplier batch size and serialize it to a string.
  onnxruntime::Model bm_model("bm_model", false, ModelMetaData(), PathString(),
                              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                              logging_manager.DefaultLogger());
  ModelTestBuilder bm_helper(bm_model.MainGraph());
  std::string bm_model_data;
  bm_model_fn(bm_helper);
  bm_helper.SetGraphOutputs();
  ASSERT_STATUS_OK(bm_model.MainGraph().Resolve());
  bm_model.ToProto().SerializeToString(&bm_model_data);

  // Run FP32 model on CPU EP and collect outputs (baseline).
  std::vector<OrtValue> cpu_bm_outputs;
  InferenceModel(bm_model_data, "bm_model_logger", {}, ExpectedEPNodeAssignment::All,
                 bm_helper.feeds_, cpu_bm_outputs);
  ASSERT_FALSE(cpu_bm_outputs.empty());

  const size_t num_outputs = cpu_bm_outputs.size();

  // Collect output values for comparison.
  std::vector<gsl::span<const float>> output_vals;
  output_vals.resize(num_outputs);

  for (size_t i = 0; i < num_outputs; i++) {
    auto& tensor = cpu_bm_outputs[i].Get<Tensor>();
    output_vals[i] = tensor.DataAsSpan<float>();
  }

  // 2. Create model with original batch size and serialize it to a string.
  onnxruntime::Model ori_model("ori_model", false, ModelMetaData(), PathString(),
                               IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                               logging_manager.DefaultLogger());
  ModelTestBuilder ori_helper(ori_model.MainGraph());
  std::string ori_model_data;
  ori_model_fn(ori_helper);
  ori_helper.SetGraphOutputs();
  ASSERT_STATUS_OK(ori_model.MainGraph().Resolve());
  ori_model.ToProto().SerializeToString(&ori_model_data);

  // 3. Run original batch size model on QNN HTP EP with batch multiplier batch size input
  const bool is_qnn_ep = true;
  TryEnableQNNSaver(const_cast<ProviderOptions&>(qnn_options));
  std::vector<OrtValue> qnn_ori_outputs;

  if (!qnn_ctx_model_path.empty()) {
    onnx::ModelProto model_proto;
    onnxruntime::Model qnn_ctx_model;
    ASSERT_STATUS_OK(qnn_ctx_model.Load(ToPathString(qnn_ctx_model_path), model_proto));
    std::string qnn_ctx_model_data;
    model_proto.SerializeToString(&qnn_ctx_model_data);
    InferenceModel(qnn_ctx_model_data, "qnn_ctx_model_logger", qnn_options,
                   expected_ep_assignment, bm_helper.feeds_, qnn_ori_outputs, is_qnn_ep, session_option_pairs);
  } else {
    // To test batch multiplier, run original batch size model using batch multiplier batch size input data.
    // Use bm_helper.feeds_ (batch multiplier size) instead of ori_helper.feeds_ (original size) for inference.
    InferenceModel(ori_model_data, "ori_model_logger", qnn_options, expected_ep_assignment,
                   bm_helper.feeds_, qnn_ori_outputs, is_qnn_ep, session_option_pairs);
  }

  // 4. Validate the outputs
  // Since HTP runs on FP16, we check whether the error between HTP with batch multiplier and ORT CPU
  // is smaller than the error between HTP without batch multiplier and ORT CPU.
  if (expected_ep_assignment != ExpectedEPNodeAssignment::None) {
    // Run batch multiplier batch size model using bathc muliplier batch size input data on QNN EP.
    std::vector<OrtValue> qnn_bm_outputs;
    InferenceModel(bm_model_data, "bm_model_logger", qnn_options, expected_ep_assignment,
                   bm_helper.feeds_, qnn_bm_outputs, is_qnn_ep, session_option_pairs);

    ASSERT_EQ(qnn_ori_outputs.size(), num_outputs);
    ASSERT_EQ(qnn_bm_outputs.size(), num_outputs);

    // Limit the error message count in case test with large data fails
    constexpr size_t max_error_count = 10;
    size_t error_count = 0;

    // Compare accuracy of ori@QNN_HTP results with bm@CPU_EP baseline
    const std::string base_output_name = "output_";
    for (size_t i = 0; i < num_outputs; i++) {
      const std::string debug_output_name = base_output_name + std::to_string(i);
      auto& qnn_ori_tensor = qnn_ori_outputs[i].Get<Tensor>();
      auto& qnn_bm_tensor = qnn_bm_outputs[i].Get<Tensor>();

      const size_t num_vals = output_vals[i].size();
      gsl::span<const float> cpu_bm_vals = output_vals[i];
      gsl::span<const float> qnn_ori_vals = qnn_ori_tensor.DataAsSpan<float>();
      gsl::span<const float> qnn_bm_vals = qnn_bm_tensor.DataAsSpan<float>();

      ASSERT_EQ(num_vals, qnn_ori_vals.size());
      ASSERT_EQ(num_vals, qnn_bm_vals.size());

      float max_qnn_ori_err = 0.0f;
      float max_qnn_bm_err = 0.0f;

      for (size_t j = 0; j < num_vals && error_count < max_error_count; j++) {
        const float expected_val = cpu_bm_vals[j];  // bm@CPU_EP val ("ground-truth")
        const float qnn_ori_val = qnn_ori_vals[j];  // ori@QNN_HTP val
        const float qnn_bm_val = qnn_bm_vals[j];

        // Calculate relative error of ori@QNN_HTP against bm@CPU_EP
        constexpr float epsilon = 1e-16f;
        const float qnn_ori_relative_err = std::fabs(expected_val - qnn_ori_val) / (std::fabs(expected_val) + epsilon);
        const float qnn_bm_relative_err = std::fabs(expected_val - qnn_bm_val) / (std::fabs(expected_val) + epsilon);

        // error between w/ and w/o batch multiplier on QNN HTP
        const float qnn_vals_err = std::fabs(qnn_ori_relative_err - qnn_bm_relative_err);
        const bool is_as_accurate_as_without_bm = qnn_ori_relative_err <= qnn_bm_relative_err;
        const bool qnn_vals_diff_within_tolerance = qnn_vals_err <= tolerance;

        const bool passed_test = is_as_accurate_as_without_bm || qnn_vals_diff_within_tolerance;
        if (!passed_test) {
          ++error_count;
        }
        EXPECT_TRUE(passed_test)
            << "Inaccuracy detected for output '" << debug_output_name
            << "', element " << j << ", tolerance=" << (tolerance * 100) << "%"
            << ".\nExpected val (bm@CPU_EP): " << expected_val
            << "\nori@QNN_HTP val: " << qnn_ori_val
            << "\nbm@QNN_HTP val: " << qnn_bm_val
            << "\nQNN HTP 'original batch size' Relative error: " << (qnn_ori_relative_err * 100) << "%"
            << "\nQNN HTP 'batch multiplier batch size' Relative error: " << (qnn_bm_relative_err * 100) << "%";

        max_qnn_ori_err = std::max(max_qnn_ori_err, qnn_ori_relative_err);
        max_qnn_bm_err = std::max(max_qnn_bm_err, qnn_bm_relative_err);
      }

      if (error_count > 0) {
        std::cerr << std::endl
                  << "[WARNING]: Output " << i
                  << " required larger tolerance to pass accuracy checks" << std::endl
                  << "Max ori relative error against bm@CPU_EP = " << (max_qnn_ori_err * 100) << "%" << std::endl
                  << "Max bm relative error against bm@CPU_EP = " << (max_qnn_bm_err * 100) << "%" << std::endl
                  << "Tolerance used = " << (tolerance * 100) << "%" << std::endl;
      }
    }
  }
}

/**
 * Tests batch multiplier accuracy by comparing QNN HTP backend (with batch multiplier)
 * against ORT CPU backend (without batch multiplier).
 *
 * @param op_type The operator type (e.g., "Conv", "MatMul")
 * @param input_defs Input definitions with original batch size
 * @param input_bm_defs Input definitions with batch multiplier batch size
 * @param attrs Operator attributes
 * @param opset_version ONNX opset version
 * @param expected_ep_assignment Expected EP node assignment
 * @param op_domain Operator domain (default: kOnnxDomain)
 * @param tolerance Relative error tolerance (default: 0.004)
 */
static void RunBatchMultiplierOpTest(
    const std::string& op_type,
    const std::vector<TestInputDef<float>>& input_defs,
    const std::vector<TestInputDef<float>>& input_bm_defs,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
    int opset_version,
    ExpectedEPNodeAssignment expected_ep_assignment,
    const std::string& op_domain = kOnnxDomain,
    float tolerance = 0.004f) {
  // Configure QNN HTP backend options
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  ProviderOptions session_options;
  session_options["session.disable_cpu_ep_fallback"] = "1";
  session_options["ep.qnn.enable_htp_batch_multiplier"] = "1";

  // Build FP32 models
  auto model_bm_fn = BuildOpTestCase<float>(op_type, input_bm_defs, {}, attrs, op_domain);
  auto model_fn = BuildOpTestCase<float>(op_type, input_defs, {}, attrs, op_domain);

  // Test FP32 batch multiplier accuracy
  TestModelBatchMultiplierAccuracy(
      model_bm_fn,
      model_fn,
      provider_options,
      opset_version,
      expected_ep_assignment,
      tolerance,
      logging::Severity::kERROR,
      "",
      session_options);
}

/**
 * Helper function to test Conv operator with different batch multiplier sizes.
 *
 * @param batch_multiplier_size The batch size to use for inference (e.g., 2, 4, 8, 16)
 */
static void TestConvBatchMultiplier(int64_t batch_multiplier_size) {
  // Constants
  constexpr int64_t kOriginalBatchSize = 1;
  constexpr int64_t kInputChannels = 1;
  constexpr int64_t kInputHeight = 5;
  constexpr int64_t kInputWidth = 5;
  constexpr int64_t kKernelSize = 3;
  constexpr size_t kWeightDataSize = kInputChannels * kInputChannels * kKernelSize * kKernelSize;
  const size_t kInputDataSize = batch_multiplier_size * kOriginalBatchSize * kInputChannels * kInputHeight * kInputWidth;

  // Generate fixed weight data to ensure consistency across model builds
  std::vector<float> weight_data(kWeightDataSize);
  std::default_random_engine weight_generator(12345);
  std::uniform_real_distribution<float> weight_distribution(-10.0f, 10.0f);
  for (auto& val : weight_data) {
    val = weight_distribution(weight_generator);
  }

  // Generate fixed input data for reproducible results
  std::vector<float> input_data(kInputDataSize);
  std::default_random_engine input_generator(6677);
  std::uniform_real_distribution<float> input_distribution(0.0f, 10.0f);
  for (auto& val : input_data) {
    val = input_distribution(input_generator);
  }

  // Create input definitions with original batch size
  std::vector<TestInputDef<float>> input_defs;
  input_defs.push_back(TestInputDef<float>(
      {kOriginalBatchSize, kInputChannels, kInputHeight, kInputWidth},
      false, 0.0f, 10.0f));  // Random data OK for compilation only
  input_defs.push_back(TestInputDef<float>(
      {kInputChannels, kInputChannels, kKernelSize, kKernelSize},
      true, weight_data));
  input_defs.push_back(TestInputDef<float>({kInputChannels}, true, {2.0f}));

  // Create input definitions with batch multiplier size
  std::vector<TestInputDef<float>> input_bm_defs;
  input_bm_defs.push_back(TestInputDef<float>(
      {batch_multiplier_size, kInputChannels, kInputHeight, kInputWidth},
      false, input_data));
  input_bm_defs.push_back(TestInputDef<float>(
      {kInputChannels, kInputChannels, kKernelSize, kKernelSize},
      true, weight_data));
  input_bm_defs.push_back(TestInputDef<float>({kInputChannels}, true, {2.0f}));

  // Configure Conv operator attributes
  std::vector<ONNX_NAMESPACE::AttributeProto> attrs;
  attrs.push_back(utils::MakeAttribute("auto_pad", "NOTSET"));
  attrs.push_back(utils::MakeAttribute("strides", std::vector<int64_t>{1, 1}));
  attrs.push_back(utils::MakeAttribute("pads", std::vector<int64_t>{0, 0, 0, 0}));
  attrs.push_back(utils::MakeAttribute("dilations", std::vector<int64_t>{1, 1}));

  RunBatchMultiplierOpTest("Conv",
                           input_defs,
                           input_bm_defs,
                           attrs,
                           21,  // opset version
                           ExpectedEPNodeAssignment::All,
                           kOnnxDomain);
}

// Test batch multiplier accuracy for Conv operator with batch size 2, 8, 128.
TEST_F(QnnHTPBackendTests, BatchMultiplier_Conv) {
  TestConvBatchMultiplier(2);
  TestConvBatchMultiplier(8);
  TestConvBatchMultiplier(128);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif
