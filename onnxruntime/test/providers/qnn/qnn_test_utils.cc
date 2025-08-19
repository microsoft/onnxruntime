// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "test/providers/qnn/qnn_test_utils.h"
#include <cassert>
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test/test_environment.h"

#include "core/platform/env_var_utils.h"
#include "core/common/span_utils.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/optimizer/graph_optimizer_registry.h"

namespace onnxruntime {
namespace test {

std::vector<float> GetFloatDataInRange(float min_val, float max_val, size_t num_elems) {
  if (num_elems == 0) {
    return {};
  }

  if (num_elems == 1) {
    return {min_val};
  }

  std::vector<float> data;
  data.reserve(num_elems);

  const float step_size = (max_val - min_val) / static_cast<float>(num_elems - 1);
  float val = min_val;
  for (size_t i = 0; i < num_elems; i++) {
    data.push_back(val);
    val += step_size;
  }

  // Ensure that max_val is included exactly (due to rounding from adding step sizes).
  data[num_elems - 1] = max_val;

  return data;
}

std::vector<float> GetSequentialFloatData(const std::vector<int64_t>& shape, float start, float step) {
  if (shape.empty()) {
    return {};
  }

  int64_t count = 1;
  for (auto dim : shape) {
    count *= dim;
  }

  std::vector<float> data;
  data.reserve(static_cast<size_t>(count));

  float val = start;
  for (int64_t i = 0; i < count; i++) {
    data.push_back(val);
    val += step;
  }

  return data;
}

TestInputDef<MLFloat16> ConvertToFP16InputDef(const TestInputDef<float>& input_def) {
  if (input_def.IsRawData()) {
    std::vector<MLFloat16> input_data_fp16;
    input_data_fp16.reserve(input_def.GetRawData().size());
    for (float f32_val : input_def.GetRawData()) {
      input_data_fp16.push_back(MLFloat16(f32_val));
    }

    return TestInputDef<MLFloat16>(input_def.GetShape(), input_def.IsInitializer(), input_data_fp16);
  } else {
    auto rand_data = input_def.GetRandomDataInfo();
    return TestInputDef<MLFloat16>(input_def.GetShape(), input_def.IsInitializer(),
                                   MLFloat16(rand_data.min), MLFloat16(rand_data.max));
  }
}

void TryEnableQNNSaver(ProviderOptions& qnn_options) {
  // Allow dumping QNN API calls to file by setting an environment variable that enables the QNN Saver backend.
  constexpr auto kEnableQNNSaverEnvironmentVariableName = "ORT_UNIT_TEST_ENABLE_QNN_SAVER";
  static std::optional<int> enable_qnn_saver = onnxruntime::ParseEnvironmentVariable<int>(
      kEnableQNNSaverEnvironmentVariableName);

  if (enable_qnn_saver.has_value() && *enable_qnn_saver != 0) {
#if defined(_WIN32)
    qnn_options["qnn_saver_path"] = "QnnSaver.dll";
#else
    qnn_options["qnn_saver_path"] = "libQnnSaver.so";
#endif  // defined(_WIN32)
  }
}

void RegisterQnnEpLibrary(RegisteredEpDeviceUniquePtr& registered_ep_device,
                          Ort::SessionOptions& session_options,
                          const std::string& registration_name,
                          const std::unordered_map<std::string, std::string>& ep_options,
                          bool simulated) {
  Ort::Env* ort_env = GetOrtEnv();
  const OrtApi& c_api = Ort::GetApi();

  std::filesystem::path library_path = "";
  if (simulated) {
    library_path =
#if _WIN32
        "onnxruntime_providers_qnn_abi_simulation.dll";
#else
        "libonnxruntime_providers_qnn_abi_simulation.so";
#endif
  } else {
    library_path =
#if _WIN32
        "onnxruntime_providers_qnn_abi.dll";
#else
        "libonnxruntime_providers_qnn_abi.so";
#endif
  }

  ASSERT_ORTSTATUS_OK(c_api.RegisterExecutionProviderLibrary(*ort_env,
                                                             registration_name.c_str(),
                                                             library_path.c_str()));

  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices;
  ASSERT_ORTSTATUS_OK(c_api.GetEpDevices(*ort_env, &ep_devices, &num_devices));

  auto target_hw_device_type = OrtHardwareDeviceType_CPU;
  if ((ep_options.find("backend_type") != ep_options.end() && ep_options.at("backend_type") == "htp") ||
      (ep_options.find("backend_path") != ep_options.end() && ep_options.at("backend_path") ==
#if _WIN32
                                                                  "QnnHtp.dll"
#else
                                                                  "libQnnHtp.so"
#endif
       )) {
    target_hw_device_type = OrtHardwareDeviceType_NPU;
  }

  auto it = std::find_if(ep_devices, ep_devices + num_devices,
                         [&c_api, &registration_name, &target_hw_device_type](const OrtEpDevice* ep_device) {
                           return (c_api.EpDevice_EpName(ep_device) == registration_name &&
                                   c_api.HardwareDevice_Type(c_api.EpDevice_Device(ep_device)) == target_hw_device_type);
                         });

  ASSERT_NE(it, ep_devices + num_devices);

  registered_ep_device = RegisteredEpDeviceUniquePtr(*it, [registration_name](const OrtEpDevice* /*ep*/) {
    Ort::GetApi().UnregisterExecutionProviderLibrary(*GetOrtEnv(), registration_name.c_str());
  });

  session_options.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(registered_ep_device.get())}, ep_options);
}

void RunQnnModelTest(const GetTestModelFn& build_test_case, ProviderOptions provider_options,
                     int opset_version, ExpectedEPNodeAssignment expected_ep_assignment,
                     float fp32_abs_err, logging::Severity log_severity, bool verify_outputs,
                     std::function<void(const Graph&)>* ep_graph_checker) {
  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = expected_ep_assignment;
  verification_params.fp32_abs_err = fp32_abs_err;
  verification_params.graph_verifier = ep_graph_checker;
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
  TryEnableQNNSaver(provider_options);
  RunAndVerifyOutputsWithEP(AsByteSpan(model_data.data(), model_data.size()), "QNN_EP_TestLogID",
                            QnnExecutionProviderWithOptions(provider_options),
                            helper.feeds_, verification_params,
                            {}, verify_outputs);

#if !BUILD_QNN_EP_STATIC_LIB
  // Run with QNN-ABI.
  std::cout << "DEBUG: ABI Test" << std::endl;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string& registration_name = "QnnAbiTestProvider";
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);

  RunAndVerifyOutputsWithEPABI(AsByteSpan(model_data.data(), model_data.size()),
                               session_options,
                               registration_name,
                               "QNN_EP_ABI_TestLogID",
                               helper.feeds_,
                               verification_params,
                               verify_outputs);
#endif  // !BUILD_QNN_EP_STATIC_LIB
}

void InferenceModel(const std::string& model_data, const char* log_id,
                    const ProviderOptions& provider_options,
                    ExpectedEPNodeAssignment expected_ep_assignment, const NameMLValMap& feeds,
                    std::vector<OrtValue>& output_vals,
                    bool is_qnn_ep,
                    const std::unordered_map<std::string, std::string>& session_option_pairs,
                    std::function<void(const Graph&)>* graph_checker) {
  SessionOptions so;
  so.session_logid = log_id;
  for (auto key_value : session_option_pairs) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(key_value.first.c_str(), key_value.second.c_str()));
  }
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  InferenceSessionWrapper session_object{so, GetEnvironment()};

  std::string provider_type = kCpuExecutionProvider;
  if (is_qnn_ep) {
    auto qnn_ep = QnnExecutionProviderWithOptions(provider_options, &so);
    provider_type = qnn_ep->Type();
    ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(qnn_ep)));
  }
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  ASSERT_STATUS_OK(session_object.Initialize());

  const auto& graph = session_object.GetGraph();

  auto ep_nodes = CountAssignedNodes(graph, provider_type);
  if (expected_ep_assignment == ExpectedEPNodeAssignment::All) {
    // Verify the entire graph is assigned to the EP
    ASSERT_EQ(ep_nodes, graph.NumberOfNodes()) << "Not all nodes were assigned to " << provider_type;
  } else if (expected_ep_assignment == ExpectedEPNodeAssignment::None) {
    ASSERT_EQ(ep_nodes, 0) << "No nodes are supposed to be assigned to " << provider_type;
  } else {
    ASSERT_GT(ep_nodes, 0) << "No nodes were assigned to " << provider_type;
  }

  if (graph_checker) {
    (*graph_checker)(graph);
  }

  const auto& outputs = graph.GetOutputs();
  std::vector<std::string> output_names;

  output_names.reserve(outputs.size());
  for (const auto* node_arg : outputs) {
    if (node_arg->Exists()) {
      output_names.push_back(node_arg->Name());
    }
  }

  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &output_vals));
}

void InferenceModelABI(const std::string& model_data,
                       const char* log_id,
                       const ProviderOptions& provider_options,
                       ExpectedEPNodeAssignment expected_ep_assignment,
                       const NameMLValMap& feeds,
                       std::vector<OrtValue>& output_vals,
                       const std::unordered_map<std::string, std::string>& session_option_pairs,
                       std::function<void(const Graph&)>* graph_checker) {
  std::cout << "DEBUG: ABI InferenceModel" << std::endl;
  RegisteredEpDeviceUniquePtr registered_ep_device;
  const std::string& registration_name = "QnnAbiTestProvider";
  Ort::SessionOptions session_options;
  RegisterQnnEpLibrary(registered_ep_device, session_options, registration_name, provider_options);

  session_options.SetLogId(log_id);
  for (auto key_value : session_option_pairs) {
    session_options.AddConfigEntry(key_value.first.c_str(), key_value.second.c_str());
  }

  Ort::RunOptions ort_run_options;
  ort_run_options.SetRunTag(log_id);

  OrtSessionWrapper ort_session(*GetOrtEnv(), model_data.data(), static_cast<int>(model_data.size()), session_options);

  // Verify node assignment.
  const auto& graph = ort_session.GetGraph();

  auto ep_nodes = CountAssignedNodes(graph, registration_name);
  if (expected_ep_assignment == ExpectedEPNodeAssignment::All) {
    ASSERT_EQ(ep_nodes, graph.NumberOfNodes()) << "Not all nodes were assigned to " << registration_name;
  } else if (expected_ep_assignment == ExpectedEPNodeAssignment::None) {
    ASSERT_EQ(ep_nodes, 0) << "No nodes are supposed to be assigned to " << registration_name;
  } else {
    ASSERT_GT(ep_nodes, 0) << "No nodes were assigned to " << registration_name;
  }

  if (graph_checker) {
    (*graph_checker)(graph);
  }

  RunWithEPABI(&ort_session, ort_run_options, feeds, output_vals);
}

NodeArg* MakeTestQDQBiasInput(ModelTestBuilder& builder, const TestInputDef<float>& bias_def, float bias_scale,
                              bool use_contrib_qdq) {
  NodeArg* bias_int32 = nullptr;

  // Bias must be int32 to be detected as a QDQ node unit.
  // We must quantize the data.
  if (bias_def.IsRandomData()) {
    // Create random initializer def that is quantized to int32
    const auto& rand_info = bias_def.GetRandomDataInfo();
    TestInputDef<int32_t> bias_int32_def(bias_def.GetShape(), bias_def.IsInitializer(),
                                         static_cast<int32_t>(rand_info.min / bias_scale),
                                         static_cast<int32_t>(rand_info.max / bias_scale));
    bias_int32 = MakeTestInput(builder, bias_int32_def);
  } else {
    assert(bias_def.IsRawData());
    // Create raw data initializer def that is quantized to int32
    const auto& bias_f32_raw = bias_def.GetRawData();
    const size_t num_elems = bias_f32_raw.size();

    std::vector<int32_t> bias_int32_raw(num_elems);
    for (size_t i = 0; i < num_elems; i++) {
      bias_int32_raw[i] = static_cast<int32_t>(bias_f32_raw[i] / bias_scale);
    }

    TestInputDef<int32_t> bias_int32_def(bias_def.GetShape(), bias_def.IsInitializer(), bias_int32_raw);
    bias_int32 = MakeTestInput(builder, bias_int32_def);
  }

  auto* bias = builder.MakeIntermediate();
  builder.AddDequantizeLinearNode<int32_t>(bias_int32, bias_scale, 0, bias, use_contrib_qdq);

  return bias;
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
      {{"backend_type", "htp"}, {"offload_graph_io_quantization", "0"}});
  GraphOptimizerRegistry graph_optimizer_registry(nullptr, nullptr, nullptr);  // as a placeholder to feed into GetCapability

  qnn_ep->SetLogger(&logger);
  auto result = qnn_ep->GetCapability(graph_viewer, kernel_lookup, graph_optimizer_registry, nullptr);

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

// Checks if Qnn Gpu backend can run a graph on the system.
// Creates a one node graph with relu op,
// then calls QNN EP's GetCapability() function
// to check if the GPU backend is available.
static BackendSupport GetGPUSupport(const onnxruntime::logging::Logger& logger) {
  onnxruntime::Model model("Check if GPU is available", false, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);

  // Build simple QDQ graph: DQ -> InstanceNormalization -> Q
  auto build_test_case = BuildOpTestCase<float, float>(
      "Relu",
      {TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f)},
      {},
      {});

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
      {{"backend_type", "gpu"}, {"offload_graph_io_quantization", "0"}});
  GraphOptimizerRegistry graph_optimizer_registry(nullptr, nullptr, nullptr);  // as a placeholder to feed into GetCapability

  qnn_ep->SetLogger(&logger);
  auto result = qnn_ep->GetCapability(graph_viewer, kernel_lookup, graph_optimizer_registry, nullptr);

  return result.empty() ? BackendSupport::UNSUPPORTED : BackendSupport::SUPPORTED;
}

void QnnGPUBackendTests::SetUp() {
  if (cached_gpu_support_ == BackendSupport::SUPPORTED) {
    return;
  }

  const auto& logger = DefaultLoggingManager().DefaultLogger();

  // Determine if GPU backend is supported only if we haven't done so before.
  if (cached_gpu_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_gpu_support_ = GetGPUSupport(logger);  // BackendSupport::SUPPORTED;
  }

  if (cached_gpu_support_ == BackendSupport::UNSUPPORTED) {
    LOGS(logger, WARNING) << "QNN GPU backend is not available! Skipping test.";
    GTEST_SKIP();
  } else if (cached_gpu_support_ == BackendSupport::SUPPORT_ERROR) {
    LOGS(logger, ERROR) << "Failed to check if QNN GPU backend is available.";
    FAIL();
  }
}

static BackendSupport GetIRSupport(const onnxruntime::logging::Logger& logger);

BackendSupport QnnHTPBackendTests::IsIRBackendSupported() const {
  const auto& logger = DefaultLoggingManager().DefaultLogger();

  if (cached_ir_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_ir_support_ = test::GetIRSupport(logger);
  }

  return cached_ir_support_;
}

// Testing helper function that calls QNN EP's GetCapability() function with a mock graph to check
// if the QNN CPU backend is available.
// TODO: Remove once the QNN CPU backend works on Windows ARM64 pipeline VM.
static BackendSupport GetCPUSupport(const onnxruntime::logging::Logger& logger, const std::string& backend_type = "cpu") {
  onnxruntime::Model model("Check if " + backend_type + " is available", false, logger);
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
      {{"backend_type", backend_type}, {"offload_graph_io_quantization", "0"}});
  GraphOptimizerRegistry graph_optimizer_registry(nullptr, nullptr, nullptr);  // as a placeholder to feed into GetCapability

  qnn_ep->SetLogger(&logger);
  auto result = qnn_ep->GetCapability(graph_viewer, kernel_lookup, graph_optimizer_registry, nullptr);

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

static BackendSupport GetIRSupport(const onnxruntime::logging::Logger& logger) {
  // QnnIr should be able to serialize any model supported by the QNN reference spec.
  // Use a model that works on QnnCpu to verify QnnIr availability.
  return GetCPUSupport(logger, "ir");
}

void QnnIRBackendTests::SetUp() {
  if (cached_ir_support_ == BackendSupport::SUPPORTED) {
    return;
  }

  const auto& logger = DefaultLoggingManager().DefaultLogger();

  // Determine if IR backend is supported only if we done so haven't before.
  if (cached_ir_support_ == BackendSupport::SUPPORT_UNKNOWN) {
    cached_ir_support_ = GetIRSupport(logger);
  }

  if (cached_ir_support_ == BackendSupport::UNSUPPORTED) {
    LOGS(logger, WARNING) << "QNN IR backend is not available! Skipping test.";
    GTEST_SKIP();
  } else if (cached_ir_support_ == BackendSupport::SUPPORT_ERROR) {
    LOGS(logger, ERROR) << "Failed to check if QNN IR backend is available.";
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

BackendSupport QnnHTPBackendTests::cached_ir_support_ = BackendSupport::SUPPORT_UNKNOWN;
BackendSupport QnnIRBackendTests::cached_ir_support_ = BackendSupport::SUPPORT_UNKNOWN;
BackendSupport QnnGPUBackendTests::cached_gpu_support_ = BackendSupport::SUPPORT_UNKNOWN;

bool ReduceOpHasAxesInput(const std::string& op_type, int opset_version) {
  static const std::unordered_map<std::string, int> opset_with_axes_as_input = {
      {"ReduceMax", 18},
      {"ReduceMin", 18},
      {"ReduceMean", 18},
      {"ReduceProd", 18},
      {"ReduceSum", 13},
      {"ReduceL2", 18},
  };

  const auto it = opset_with_axes_as_input.find(op_type);

  return (it != opset_with_axes_as_input.cend()) && (it->second <= opset_version);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
