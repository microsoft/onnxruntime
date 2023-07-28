// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "test/providers/qnn/qnn_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test/test_environment.h"

#include "core/graph/graph.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {
namespace test {

void RunQnnModelTest(const GetTestModelFn& build_test_case, const ProviderOptions& provider_options,
                     int opset_version, ExpectedEPNodeAssignment expected_ep_assignment,
                     float fp32_abs_err, logging::Severity log_severity) {
  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = expected_ep_assignment;
  verification_params.fp32_abs_err = fp32_abs_err;
  // Add kMSDomain to cover contrib op like Gelu
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(log_severity);

  onnxruntime::Model model("QNN_EP_TestModel", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  build_test_case(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  RunAndVerifyOutputsWithEP(model_data, "QNN_EP_TestLogID",
                            QnnExecutionProviderWithOptions(provider_options),
                            helper.feeds_, verification_params);
}

// Mock IKernelLookup class passed to QNN EP's GetCapability() function in order to
// determine if the HTP backend is supported on specific platforms (e.g., Windows ARM64).
// TODO: Remove once HTP can be emulated on Windows ARM64.
class MockKernelLookup : public onnxruntime::IExecutionProvider::IKernelLookup {
 public:
  const KernelCreateInfo* LookUpKernel(const Node& /* node */) const {
    // Do nothing.
    return nullptr;
  }
};

// Testing helper function that calls QNN EP's GetCapability() function with a mock graph to check
// if the HTP backend is available.
// TODO: Remove once HTP can be emulated on Windows ARM64.
static BackendSupport GetHTPSupport(const onnxruntime::logging::Logger& logger) {
  onnxruntime::Model model("Check if HTP is available", false, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);

  // Build simple QDQ graph: DQ -> InstanceNormalization -> Q
  GetQDQTestCaseFn build_test_case = [](ModelTestBuilder& builder) {
    const uint8_t quant_zero_point = 0;
    const float quant_scale = 1.0f;

    auto* dq_scale_output = builder.MakeIntermediate();
    auto* scale = builder.MakeInitializer<uint8_t>({2}, std::vector<uint8_t>{1, 2});
    builder.AddDequantizeLinearNode<uint8_t>(scale, quant_scale, quant_zero_point, dq_scale_output);

    // Add bias (initializer) -> DQ ->
    auto* dq_bias_output = builder.MakeIntermediate();
    auto* bias = builder.MakeInitializer<int32_t>({2}, std::vector<int32_t>{1, 1});
    builder.AddDequantizeLinearNode<int32_t>(bias, 1.0f, 0, dq_bias_output);

    // Add input_u8 -> DQ ->
    auto* input_u8 = builder.MakeInput<uint8_t>({1, 2, 3}, std::vector<uint8_t>{1, 2, 3, 4, 5, 6});
    auto* dq_input_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<uint8_t>(input_u8, quant_scale, quant_zero_point, dq_input_output);

    // Add dq_input_output -> InstanceNormalization ->
    auto* instance_norm_output = builder.MakeIntermediate();
    builder.AddNode("InstanceNormalization", {dq_input_output, dq_scale_output, dq_bias_output},
                    {instance_norm_output});

    // Add instance_norm_output -> Q -> output_u8
    auto* output_u8 = builder.MakeOutput();
    builder.AddQuantizeLinearNode<uint8_t>(instance_norm_output, quant_scale, quant_zero_point, output_u8);
  };

  build_test_case(helper);
  helper.SetGraphOutputs();
  auto status = model.MainGraph().Resolve();

  if (!status.IsOK()) {
    return BackendSupport::SUPPORT_ERROR;
  }

  // Create QNN EP and call GetCapability().
  MockKernelLookup kernel_lookup;
  onnxruntime::GraphViewer graph_viewer(graph);
  std::unique_ptr<onnxruntime::IExecutionProvider> qnn_ep = QnnExecutionProviderWithOptions(
      {{"backend_path", "QnnHtp.dll"}});

  qnn_ep->SetLogger(&logger);
  auto result = qnn_ep->GetCapability(graph_viewer, kernel_lookup);

  return result.empty() ? BackendSupport::UNSUPPORTED : BackendSupport::SUPPORTED;
}

void QnnHTPBackendTests::SetUp() {
  if (cached_htp_support_ == BackendSupport::SUPPORTED) {
    return;
  }

  const auto& logger = DefaultLoggingManager().DefaultLogger();

  // Determine if HTP backend is supported only if we done so haven't before.
  if (cached_htp_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_htp_support_ = GetHTPSupport(logger);
  }

  if (cached_htp_support_ == BackendSupport::UNSUPPORTED) {
    LOGS(logger, WARNING) << "QNN HTP backend is not available! Skipping test.";
    GTEST_SKIP();
  } else if (cached_htp_support_ == BackendSupport::SUPPORT_ERROR) {
    LOGS(logger, ERROR) << "Failed to check if QNN HTP backend is available.";
    FAIL();
  }
}

// Testing helper function that calls QNN EP's GetCapability() function with a mock graph to check
// if the QNN CPU backend is available.
// TODO: Remove once the QNN CPU backend works on Windows ARM64 pipeline VM.
static BackendSupport GetCPUSupport(const onnxruntime::logging::Logger& logger) {
  onnxruntime::Model model("Check if CPU is available", false, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);

  auto get_test_model_func = [](const std::vector<int64_t>& input_shape) -> GetTestModelFn {
    return [input_shape](ModelTestBuilder& builder) {
      const int64_t num_channels = input_shape[1];

      auto* scale = builder.MakeInitializer<float>({num_channels}, 0.0f, 1.0f);
      auto* bias = builder.MakeInitializer<float>({num_channels}, 0.0f, 4.0f);
      auto* input_arg = builder.MakeInput<float>(input_shape, 0.0f, 10.0f);
      auto* instance_norm_output = builder.MakeOutput();
      builder.AddNode("InstanceNormalization", {input_arg, scale, bias}, {instance_norm_output});
    };
  };

  // Build simple graph with a InstanceNormalization op.
  GetQDQTestCaseFn build_test_case = get_test_model_func({1, 2, 3, 3});
  build_test_case(helper);
  helper.SetGraphOutputs();
  auto status = model.MainGraph().Resolve();

  if (!status.IsOK()) {
    return BackendSupport::SUPPORT_ERROR;
  }

  // Create QNN EP and call GetCapability().
  MockKernelLookup kernel_lookup;
  onnxruntime::GraphViewer graph_viewer(graph);
  std::unique_ptr<onnxruntime::IExecutionProvider> qnn_ep = QnnExecutionProviderWithOptions(
      {{"backend_path", "QnnCpu.dll"}});

  qnn_ep->SetLogger(&logger);
  auto result = qnn_ep->GetCapability(graph_viewer, kernel_lookup);

  return result.empty() ? BackendSupport::UNSUPPORTED : BackendSupport::SUPPORTED;
}

void QnnCPUBackendTests::SetUp() {
  if (cached_cpu_support_ == BackendSupport::SUPPORTED) {
    return;
  }

  const auto& logger = DefaultLoggingManager().DefaultLogger();

  // Determine if CPU backend is supported only if we done so haven't before.
  if (cached_cpu_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_cpu_support_ = GetCPUSupport(logger);
  }

  if (cached_cpu_support_ == BackendSupport::UNSUPPORTED) {
    LOGS(logger, WARNING) << "QNN CPU backend is not available! Skipping test.";
    GTEST_SKIP();
  } else if (cached_cpu_support_ == BackendSupport::SUPPORT_ERROR) {
    LOGS(logger, ERROR) << "Failed to check if QNN CPU backend is available.";
    FAIL();
  }
}

#if defined(_WIN32)
// TODO: Remove or set to SUPPORTED once HTP emulation is supported on win arm64.
BackendSupport QnnHTPBackendTests::cached_htp_support_ = BackendSupport::SUPPORT_UNKNOWN;

// TODO: Remove or set to SUPPORTED once CPU backend works on win arm64 (pipeline VM).
BackendSupport QnnCPUBackendTests::cached_cpu_support_ = BackendSupport::SUPPORT_UNKNOWN;
#else
BackendSupport QnnHTPBackendTests::cached_htp_support_ = BackendSupport::SUPPORTED;
BackendSupport QnnCPUBackendTests::cached_cpu_support_ = BackendSupport::SUPPORTED;
#endif  // defined(_WIN32)

bool ReduceOpHasAxesInput(const std::string& op_type, int opset_version) {
  static const std::unordered_map<std::string, int> opset_with_axes_as_input = {
      {"ReduceMax", 18},
      {"ReduceMin", 18},
      {"ReduceMean", 18},
      {"ReduceProd", 18},
      {"ReduceSum", 13},
  };

  const auto it = opset_with_axes_as_input.find(op_type);

  return (it != opset_with_axes_as_input.cend()) && (it->second <= opset_version);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)