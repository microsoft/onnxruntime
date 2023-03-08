// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"

#include "test/util/include/test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_utils.h"

#if !defined(ORT_MINIMAL_BUILD)
// if this is a full build we need the provider test utils
#include "test/optimizer/qdq_test_utils.h"
#endif

#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

// in test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

// test uses ONNX model so can't be run in a minimal build.
// TODO: When we need QNN in a minimal build we should add an ORT format version of the model
#if !defined(ORT_MINIMAL_BUILD)

// Tests that the QNN EP is registered when added via the public C++ API.
// Loads a simple ONNX model that adds floats.
TEST(QnnEP, TestAddEpUsingPublicApi) {
  {
    // C++ API test
    Ort::SessionOptions so;
    onnxruntime::ProviderOptions options;

#if defined(_WIN32)
    options["backend_path"] = "QnnCpu.dll";
#else
    options["backend_path"] = "libQnnCpu.so";
#endif

    so.AppendExecutionProvider("QNN", options);

    const ORTCHAR_T* ort_model_path = ORT_MODEL_FOLDER "constant_floats.onnx";
    Ort::Session session(*ort_env, ort_model_path, so);

    // Access the underlying InferenceSession.
    const OrtSession* ort_session = session;
    const InferenceSession* s = reinterpret_cast<const InferenceSession*>(ort_session);

    bool have_qnn_ep = false;

    for (const auto& provider : s->GetRegisteredProviderTypes()) {
      if (provider == kQnnExecutionProvider) {
        have_qnn_ep = true;
        break;
      }
    }

    ASSERT_TRUE(have_qnn_ep) << "QNN EP was not found in registered providers for session.";
  }
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

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

enum HTPSupport {
  HTP_SUPPORT_UNKNOWN = 0,
  HTP_UNSUPPORTED,
  HTP_SUPPORTED,
  HTP_SUPPORT_ERROR,
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

// Testing fixture class for tests that require the HTP backend. Checks if HTP is available before the test begins.
// The test is skipped if HTP is unavailable (may occur on Windows ARM64).
// TODO: Remove once HTP can be emulated on Windows ARM64.
class QnnHTPBackendTests : public ::testing::Test {
 protected:
  void SetUp() override {
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

  static HTPSupport cached_htp_support_;  // Set by the first test using this fixture.
};

#if defined(_WIN32)
HTPSupport QnnHTPBackendTests::cached_htp_support_ = HTPSupport::HTP_SUPPORT_UNKNOWN;
#else
HTPSupport QnnHTPBackendTests::cached_htp_support_ = HTPSupport::HTP_SUPPORTED;
#endif  // defined(_WIN32)

// Testing helper function that runs a caller-provided QDQ graph (build_test_case) to allow the caller to
// 1) test which nodes are assigned to an EP, and 2) check that the inference output matches with the CPU EP.
static void RunModelTest(const GetQDQTestCaseFn& build_test_case, const char* test_description,
                         const ProviderOptions& provider_options,
                         const EPVerificationParams& params = EPVerificationParams()) {
  onnxruntime::Model model(test_description, false, DefaultLoggingManager().DefaultLogger());
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
                            helper.feeds_, params);
}

// Check that QNN compiles DQ -> Conv -> Q as a single unit.
TEST_F(QnnHTPBackendTests, TestQDQConvU8U8) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> Conv -> Q node unit assigned to QNN EP";
  };

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;

  RunModelTest(BuildQDQConvTestCase<uint8_t /* InputType */,
                                    uint8_t /* WeightType */,
                                    int32_t /* BiasType */,
                                    uint8_t /* OutputType */>(
                   {1, 1, 5, 5} /* input_shape */,
                   {1, 1, 3, 3} /* weights_shape */),
               "qnn_qdq_test_graph_conv_u8u8",
               provider_options,
               verification_params);  // two transpose nodes would be added before and after
}

// Check that QNN compiles DQ -> InstanceNormalization -> Q as a single unit.
// Use an input of rank 4.
TEST_F(QnnHTPBackendTests, TestQDQInstanceNormU8) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> InstanceNormalization -> Q node unit assigned to QNN EP";
  };

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  RunModelTest(BuildQDQInstanceNormTestCase<uint8_t, uint8_t, int32_t>(
                   {1, 2, 3, 3} /* input_shape */,
                   1e-05f       /* epsilon */),
               "qnn_qdq_test_graph_instance_norm_u8",
               provider_options,
               verification_params);
}

// Check that QNN compiles DQ -> InstanceNormalization -> Q as a single unit.
// Use an input of rank 3.
TEST_F(QnnHTPBackendTests, TestQDQInstanceNormU8Rank3) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  std::function<void(const Graph&)> graph_verify = [](const Graph& graph) -> void {
    ASSERT_EQ(graph.NumberOfNodes(), 1) << "DQ -> InstanceNormalization -> Q node unit assigned to QNN EP";
  };

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::All;  // All graph nodes should be assigned to QNN
  verification_params.graph_verifier = &graph_verify;

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  RunModelTest(BuildQDQInstanceNormTestCase<uint8_t, uint8_t, int32_t>(
                   {1, 2, 3} /* input_shape */,
                   1e-05f    /* epsilon */),
               "qnn_qdq_test_graph_instance_norm_u8_rank3",
               provider_options,
               verification_params);
}

// Check that QNN InstanceNorm operator does not handle inputs with rank > 4.
TEST_F(QnnHTPBackendTests, TestQDQInstanceNormU8Rank5) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  EPVerificationParams verification_params;
  verification_params.ep_node_assignment = ExpectedEPNodeAssignment::None;  // No graph nodes should be assigned to QNN

  // Runs model with DQ-> InstanceNorm -> Q and compares the outputs of the CPU and QNN EPs.
  RunModelTest(BuildQDQInstanceNormTestCase<uint8_t, uint8_t, int32_t>(
                   {1, 2, 3, 3, 3} /* input_shape */,
                   1e-05f          /* epsilon */),
               "qnn_qdq_test_graph_instance_norm_u8_rank5",
               provider_options,
               verification_params);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace test
}  // namespace onnxruntime
