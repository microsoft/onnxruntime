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
                     int opset_version, ExpectedEPNodeAssignment expected_ep_assignment, int num_nodes_in_ep,
                     const char* test_description, float fp32_abs_err) {
  std::function<void(const Graph&)> graph_verify = [num_nodes_in_ep, test_description](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), num_nodes_in_ep) << test_description;
  };

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = expected_ep_assignment;
  verification_params.graph_verifier = &graph_verify;
  verification_params.fp32_abs_err = fp32_abs_err;
  const std::unordered_map<std::string, int> domain_to_version = {{"", opset_version}};

  onnxruntime::Model model(test_description, false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  build_test_case(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  RunAndVerifyOutputsWithEP(model_data, "QnnEP.TestQDQModel",
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
static HTPSupport GetHTPSupport(const onnxruntime::logging::Logger& logger) {
  onnxruntime::Model model("Check if HTP is available", false, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);

  // Build simple QDQ graph: DQ -> InstanceNormalization -> Q
  GetQDQTestCaseFn build_test_case = BuildQDQInstanceNormTestCase<uint8_t, uint8_t, int32_t>({1, 2, 3, 3}, 1e-05f);
  build_test_case(helper);
  helper.SetGraphOutputs();
  auto status = model.MainGraph().Resolve();

  if (!status.IsOK()) {
    return HTPSupport::HTP_SUPPORT_ERROR;
  }

  // Create QNN EP and call GetCapability().
  MockKernelLookup kernel_lookup;
  onnxruntime::GraphViewer graph_viewer(graph);
  std::unique_ptr<onnxruntime::IExecutionProvider> qnn_ep = QnnExecutionProviderWithOptions(
      {{"backend_path", "QnnHtp.dll"}});

  qnn_ep->SetLogger(&logger);
  auto result = qnn_ep->GetCapability(graph_viewer, kernel_lookup);

  return result.empty() ? HTPSupport::HTP_UNSUPPORTED : HTPSupport::HTP_SUPPORTED;
}

void QnnHTPBackendTests::SetUp() {
  if (cached_htp_support_ == HTPSupport::HTP_SUPPORTED) {
    return;
  }

  const auto& logger = DefaultLoggingManager().DefaultLogger();

  // Determine if HTP backend is supported only if we done so haven't before.
  if (cached_htp_support_ == HTPSupport::HTP_SUPPORT_UNKNOWN) {
    cached_htp_support_ = GetHTPSupport(logger);
  }

  if (cached_htp_support_ == HTPSupport::HTP_UNSUPPORTED) {
    LOGS(logger, WARNING) << "QNN HTP backend is not available! Skipping test.";
    GTEST_SKIP();
  } else if (cached_htp_support_ == HTPSupport::HTP_SUPPORT_ERROR) {
    LOGS(logger, ERROR) << "Failed to check if QNN HTP backend is available.";
    FAIL();
  }
}

#if defined(_WIN32)
// TODO: Remove or set to HTP_SUPPORTED once HTP emulation is supported on win arm64.
HTPSupport QnnHTPBackendTests::cached_htp_support_ = HTPSupport::HTP_SUPPORT_UNKNOWN;
#else
HTPSupport QnnHTPBackendTests::cached_htp_support_ = HTPSupport::HTP_SUPPORTED;
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