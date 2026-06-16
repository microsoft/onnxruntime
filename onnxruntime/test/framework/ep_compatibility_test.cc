// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <climits>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/framework/execution_provider.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/utils.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/abi_session_options_impl.h"
#include "core/framework/error_code_helper.h"
#include "dummy_provider.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/test_environment.h"
#include "test/providers/provider_test_utils.h"

using namespace onnxruntime;
using namespace onnxruntime::test;

namespace {

// Test execution provider that extends IExecutionProvider with compatibility string functionality
class TestCompatibilityExecutionProvider : public IExecutionProvider {
 public:
  static constexpr const char* kTestCompatibilityExecutionProviderType = "TestCompatibilityExecutionProvider";

  TestCompatibilityExecutionProvider() : IExecutionProvider(kTestCompatibilityExecutionProviderType) {
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    return std::make_shared<KernelRegistry>();
  }

  std::vector<AllocatorPtr> CreatePreferredAllocators() override {
    return {};
  }

  // Configurable mock behavior
  void SetMockCompatibilityString(const std::string& str) {
    mock_compatibility_string_ = str;
  }

  void SetMockCompatibilityStatus(OrtCompiledModelCompatibility status) {
    mock_compatibility_status_ = status;
  }

  void SetShouldFailValidation(bool should_fail) {
    should_fail_validation_ = should_fail;
  }

  // Override compatibility methods
  std::string GetCompiledModelCompatibilityInfo(const onnxruntime::GraphViewer& graph_viewer) const override {
    ORT_UNUSED_PARAMETER(graph_viewer);
    return mock_compatibility_string_;
  }

  common::Status ValidateCompiledModelCompatibilityInfo(const std::string& compatibility_info,
                                                        OrtCompiledModelCompatibility& model_compatibility) const override {
    if (should_fail_validation_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Mock validation failure");
    }

    // Simple validation logic for testing
    // If the mock status is explicitly set to NOT_APPLICABLE, always return that
    if (mock_compatibility_status_ == OrtCompiledModelCompatibility_EP_NOT_APPLICABLE) {
      model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    } else if (compatibility_info.empty()) {
      model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    } else if (compatibility_info == mock_compatibility_string_) {
      model_compatibility = mock_compatibility_status_;
    } else {
      model_compatibility = OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }

    return Status::OK();
  }

 private:
  std::string mock_compatibility_string_ = "default_test_compatibility_v1.0";
  OrtCompiledModelCompatibility mock_compatibility_status_ = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  bool should_fail_validation_ = false;
};

// Test execution provider that tracks whether GetCapability is called.
// This is used to verify that early validation fails BEFORE Initialize() does expensive work.
class TestEarlyValidationExecutionProvider : public IExecutionProvider {
 public:
  static constexpr const char* kTestEarlyValidationExecutionProviderType = "TestEarlyValidationExecutionProvider";

  TestEarlyValidationExecutionProvider() : IExecutionProvider(kTestEarlyValidationExecutionProviderType) {
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    return std::make_shared<KernelRegistry>();
  }

  std::vector<AllocatorPtr> CreatePreferredAllocators() override {
    return {};
  }

  // Override GetCapability to track if it's called (happens during Initialize())
  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph_viewer,
      const IKernelLookup& kernel_lookup,
      const GraphOptimizerRegistry& graph_optimizer_registry,
      IResourceAccountant* resource_accountant = nullptr) const override {
    ORT_UNUSED_PARAMETER(graph_viewer);
    ORT_UNUSED_PARAMETER(kernel_lookup);
    ORT_UNUSED_PARAMETER(graph_optimizer_registry);
    ORT_UNUSED_PARAMETER(resource_accountant);
    get_capability_called_ = true;
    return {};  // Return empty - we don't actually want to handle any nodes
  }

  // Configurable mock behavior for validation
  void SetMockCompatibilityStatus(OrtCompiledModelCompatibility status) {
    mock_compatibility_status_ = status;
  }

  common::Status ValidateCompiledModelCompatibilityInfo(const std::string& compatibility_info,
                                                        OrtCompiledModelCompatibility& model_compatibility) const override {
    ORT_UNUSED_PARAMETER(compatibility_info);
    model_compatibility = mock_compatibility_status_;
    return Status::OK();
  }

  // Query whether GetCapability was called
  bool WasGetCapabilityCalled() const {
    return get_capability_called_;
  }

  void ResetGetCapabilityCalled() {
    get_capability_called_ = false;
  }

 private:
  OrtCompiledModelCompatibility mock_compatibility_status_ = OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  mutable bool get_capability_called_ = false;
};

// Helper class to create test models
class ModelBuilderWithCompatibility {
 public:
  static std::unique_ptr<Model> CreateSimpleTestModel() {
    // Create a simple model with a single Add operation
    std::unordered_map<std::string, int> domain_to_version;
    domain_to_version[onnxruntime::kOnnxDomain] = 7;

    auto p_model = std::make_unique<Model>("test_model", true, ModelMetaData(), PathString(),
                                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
                                           std::vector<ONNX_NAMESPACE::FunctionProto>(),
                                           DefaultLoggingManager().DefaultLogger());

    onnxruntime::Graph& graph = p_model->MainGraph();

    // Define tensor type
    ONNX_NAMESPACE::TypeProto tensor_float;
    tensor_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
    tensor_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

    // Create input and output node args
    auto& input_arg_a = graph.GetOrCreateNodeArg("A", &tensor_float);
    auto& input_arg_b = graph.GetOrCreateNodeArg("B", &tensor_float);
    auto& output_arg = graph.GetOrCreateNodeArg("C", &tensor_float);

    // Create Add node
    std::vector<onnxruntime::NodeArg*> input_defs = {&input_arg_a, &input_arg_b};
    std::vector<onnxruntime::NodeArg*> output_defs = {&output_arg};
    graph.AddNode("add_node", "Add", "Add two tensors", input_defs, output_defs, nullptr, onnxruntime::kOnnxDomain);

    auto status = graph.Resolve();
    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

    return p_model;
  }

  static std::unique_ptr<Model> CreateModelWithCompatibilityMetadata(
      const std::map<std::string, std::string>& ep_compatibility_info) {
    auto model = CreateSimpleTestModel();

    // Add compatibility metadata
    auto& metadata = model->MetaData();
    for (const auto& [ep_type, compatibility_string] : ep_compatibility_info) {
      std::string metadata_key = std::string(kOrtModelMetadata_EpCompatibilityInfoPrefix) + ep_type;
      metadata[metadata_key] = compatibility_string;
    }

    return model;
  }
};

// Helper class to create test sessions
class SessionBuilderWithCompatibility {
 public:
  static std::unique_ptr<InferenceSession> CreateTestSession(std::unique_ptr<Model> model, bool fail_on_suboptimal = false) {
    SessionOptions so;
    so.session_logid = "EpCompatibilityTest";
    so.session_log_verbosity_level = 1;

    if (fail_on_suboptimal) {
      EXPECT_TRUE(so.config_options.AddConfigEntry(kOrtSessionOptionsFailOnSuboptimalCompiledModel, "1").IsOK());
    }

    // Convert Model to ModelProto and serialize
    auto model_proto = model->ToProto();
    std::string model_data;
    EXPECT_TRUE(model_proto.SerializeToString(&model_data));
    std::stringstream model_stream(model_data);

    // Create session with basic constructor
    auto session = std::make_unique<InferenceSession>(so, GetEnvironment());

    // Load the model from the stream and validate the status
    auto load_status = session->Load(model_stream);
    EXPECT_TRUE(load_status.IsOK()) << "Failed to load model: " << load_status.ErrorMessage();

    return session;
  }
};

// Helper function to initialize session using the proper validation pathway
Status InitializeSessionWithValidation(InferenceSession& session) {
  // Create OrtSessionOptions from the session's SessionOptions to use the proper initialization path
  OrtSessionOptions ort_session_options;
  ort_session_options.value = session.GetSessionOptions();

  // Call the InitializeSession function from utils.cc which includes validation
  OrtStatus* ort_status = InitializeSession(&ort_session_options, session, nullptr);

  // Convert OrtStatus to Status using the proper helper function
  return ToStatusAndRelease(ort_status);
}

}  // anonymous namespace

class EpCompatibilityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_model_ = ModelBuilderWithCompatibility::CreateSimpleTestModel();
  }

 protected:
  std::unique_ptr<Model> test_model_;
};

// Test basic compatibility string generation during compilation
TEST_F(EpCompatibilityTest, TestCompatibilityStringGeneration) {
  const std::string expected_compatibility_string = "test_ep_v1.0_compatibility_data";

  auto test_ep = std::make_unique<TestCompatibilityExecutionProvider>();
  test_ep->SetMockCompatibilityString(expected_compatibility_string);

  auto session = SessionBuilderWithCompatibility::CreateTestSession(std::move(test_model_));
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::move(test_ep)));
  ASSERT_STATUS_OK(InitializeSessionWithValidation(*session));

  // Note: In the actual implementation, we would need to trigger EP context model creation
  // to see the compatibility strings stored. For now, this tests that the methods are called
  // without error during session initialization.
}

// Test compatibility string storage in model metadata
TEST_F(EpCompatibilityTest, TestCompatibilityStringStorage) {
  const std::string ep_type = "TestCompatibilityExecutionProvider";
  const std::string expected_compatibility_string = "stored_compatibility_v2.0";

  // Create model with pre-populated compatibility metadata
  std::map<std::string, std::string> compatibility_info = {
      {ep_type, expected_compatibility_string}};
  auto model_with_metadata = ModelBuilderWithCompatibility::CreateModelWithCompatibilityMetadata(compatibility_info);

  // Verify metadata was stored correctly
  const auto& metadata = model_with_metadata->MetaData();
  std::string expected_key = std::string(kOrtModelMetadata_EpCompatibilityInfoPrefix) + ep_type;

  auto it = metadata.find(expected_key);
  ASSERT_NE(it, metadata.end()) << "Expected compatibility metadata key not found: " << expected_key;
  EXPECT_EQ(it->second, expected_compatibility_string);
}

// Test multiple EPs generating different compatibility strings
TEST_F(EpCompatibilityTest, TestMultipleEpCompatibilityStrings) {
  std::map<std::string, std::string> compatibility_info = {
      {"EP_A", "ep_a_compatibility_v1.0"},
      {"EP_B", "ep_b_compatibility_v2.1"},
      {"EP_C", "ep_c_compatibility_v1.5"}};

  auto model_with_metadata = ModelBuilderWithCompatibility::CreateModelWithCompatibilityMetadata(compatibility_info);

  // Verify all compatibility strings are stored
  const auto& metadata = model_with_metadata->MetaData();
  for (const auto& [ep_type, expected_string] : compatibility_info) {
    std::string expected_key = std::string(kOrtModelMetadata_EpCompatibilityInfoPrefix) + ep_type;
    auto it = metadata.find(expected_key);
    ASSERT_NE(it, metadata.end()) << "Expected compatibility metadata key not found: " << expected_key;
    EXPECT_EQ(it->second, expected_string);
  }
}

// Test empty compatibility string handling
TEST_F(EpCompatibilityTest, TestEmptyCompatibilityString) {
  auto test_ep = std::make_unique<TestCompatibilityExecutionProvider>();
  test_ep->SetMockCompatibilityString("");  // Empty string

  auto session = SessionBuilderWithCompatibility::CreateTestSession(std::move(test_model_));
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::move(test_ep)));
  ASSERT_STATUS_OK(InitializeSessionWithValidation(*session));  // Should succeed even with empty compatibility string
}

// Test compatibility validation with optimal status
TEST_F(EpCompatibilityTest, TestCompatibilityValidation_Optimal) {
  const std::string ep_type = "TestCompatibilityExecutionProvider";
  const std::string compatibility_string = "optimal_compatibility_v1.0";

  auto test_ep = std::make_unique<TestCompatibilityExecutionProvider>();
  test_ep->SetMockCompatibilityString(compatibility_string);
  test_ep->SetMockCompatibilityStatus(OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL);

  // Create model with matching compatibility metadata
  std::map<std::string, std::string> compatibility_info = {{ep_type, compatibility_string}};
  auto model_with_metadata = ModelBuilderWithCompatibility::CreateModelWithCompatibilityMetadata(compatibility_info);

  auto session = SessionBuilderWithCompatibility::CreateTestSession(std::move(model_with_metadata));
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::move(test_ep)));
  ASSERT_STATUS_OK(InitializeSessionWithValidation(*session));  // Should succeed with optimal compatibility
}

// Test compatibility validation with suboptimal status (default session settings)
TEST_F(EpCompatibilityTest, TestCompatibilityValidation_Suboptimal_DefaultSettings) {
  const std::string ep_type = "TestCompatibilityExecutionProvider";
  const std::string compatibility_string = "suboptimal_compatibility_v1.0";

  auto test_ep = std::make_unique<TestCompatibilityExecutionProvider>();
  test_ep->SetMockCompatibilityString(compatibility_string);
  test_ep->SetMockCompatibilityStatus(OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION);

  std::map<std::string, std::string> compatibility_info = {{ep_type, compatibility_string}};
  auto model_with_metadata = ModelBuilderWithCompatibility::CreateModelWithCompatibilityMetadata(compatibility_info);

  auto session = SessionBuilderWithCompatibility::CreateTestSession(std::move(model_with_metadata), false);  // Don't fail on suboptimal
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::move(test_ep)));
  ASSERT_STATUS_OK(InitializeSessionWithValidation(*session));  // Should succeed by default with suboptimal compatibility
}

// Test compatibility validation with suboptimal status (fail on suboptimal enabled)
TEST_F(EpCompatibilityTest, TestCompatibilityValidation_Suboptimal_FailEnabled) {
  const std::string ep_type = "TestCompatibilityExecutionProvider";
  const std::string compatibility_string = "suboptimal_compatibility_v1.0";

  auto test_ep = std::make_unique<TestCompatibilityExecutionProvider>();
  test_ep->SetMockCompatibilityString(compatibility_string);
  test_ep->SetMockCompatibilityStatus(OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION);

  std::map<std::string, std::string> compatibility_info = {{ep_type, compatibility_string}};
  auto model_with_metadata = ModelBuilderWithCompatibility::CreateModelWithCompatibilityMetadata(compatibility_info);

  auto session = SessionBuilderWithCompatibility::CreateTestSession(std::move(model_with_metadata), true);  // Fail on suboptimal
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::move(test_ep)));

  // Should fail during initialization due to suboptimal compatibility
  auto status = InitializeSessionWithValidation(*session);
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("suboptimal"));
}

// Test compatibility validation with unsupported status
TEST_F(EpCompatibilityTest, TestCompatibilityValidation_Unsupported) {
  const std::string ep_type = "TestCompatibilityExecutionProvider";
  const std::string stored_compatibility_string = "old_compatibility_v1.0";
  const std::string current_compatibility_string = "new_compatibility_v2.0";

  auto test_ep = std::make_unique<TestCompatibilityExecutionProvider>();
  test_ep->SetMockCompatibilityString(current_compatibility_string);
  test_ep->SetMockCompatibilityStatus(OrtCompiledModelCompatibility_EP_UNSUPPORTED);

  // Model has old compatibility string, EP has new one -> unsupported
  std::map<std::string, std::string> compatibility_info = {{ep_type, stored_compatibility_string}};
  auto model_with_metadata = ModelBuilderWithCompatibility::CreateModelWithCompatibilityMetadata(compatibility_info);

  auto session = SessionBuilderWithCompatibility::CreateTestSession(std::move(model_with_metadata), false);  // Even with fail_on_suboptimal=false
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::move(test_ep)));

  // Should fail during initialization due to unsupported compatibility
  auto status = InitializeSessionWithValidation(*session);
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("not supported"));
}

// Test compatibility validation with not applicable status
TEST_F(EpCompatibilityTest, TestCompatibilityValidation_NotApplicable) {
  const std::string ep_type = "TestCompatibilityExecutionProvider";

  auto test_ep = std::make_unique<TestCompatibilityExecutionProvider>();
  test_ep->SetMockCompatibilityString("");  // Empty compatibility string
  test_ep->SetMockCompatibilityStatus(OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);

  // Model has some compatibility string, but EP returns not applicable
  std::map<std::string, std::string> compatibility_info = {{ep_type, "some_compatibility_string"}};
  auto model_with_metadata = ModelBuilderWithCompatibility::CreateModelWithCompatibilityMetadata(compatibility_info);

  auto session = SessionBuilderWithCompatibility::CreateTestSession(std::move(model_with_metadata));
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::move(test_ep)));
  ASSERT_STATUS_OK(InitializeSessionWithValidation(*session));  // Should succeed with not applicable status
}

// Test missing compatibility info in model metadata
TEST_F(EpCompatibilityTest, TestMissingCompatibilityInfo) {
  auto test_ep = std::make_unique<TestCompatibilityExecutionProvider>();
  test_ep->SetMockCompatibilityString("some_compatibility_string");

  // Use model without any compatibility metadata
  auto session = SessionBuilderWithCompatibility::CreateTestSession(std::move(test_model_));
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::move(test_ep)));
  ASSERT_STATUS_OK(InitializeSessionWithValidation(*session));  // Should succeed when no compatibility info is present
}

// Test EP validation failure
TEST_F(EpCompatibilityTest, TestEpValidationFailure) {
  const std::string ep_type = "TestCompatibilityExecutionProvider";
  const std::string compatibility_string = "test_compatibility_v1.0";

  auto test_ep = std::make_unique<TestCompatibilityExecutionProvider>();
  test_ep->SetMockCompatibilityString(compatibility_string);
  test_ep->SetShouldFailValidation(true);  // Force validation failure

  std::map<std::string, std::string> compatibility_info = {{ep_type, compatibility_string}};
  auto model_with_metadata = ModelBuilderWithCompatibility::CreateModelWithCompatibilityMetadata(compatibility_info);

  auto session = SessionBuilderWithCompatibility::CreateTestSession(std::move(model_with_metadata));
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::move(test_ep)));

  // Should handle EP validation failure gracefully
  auto status = InitializeSessionWithValidation(*session);
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Mock validation failure"));
}

// Test that early validation optimization works: when a model is incompatible,
// validation should fail BEFORE Initialize() performs expensive graph partitioning.
// We verify this by checking that GetCapability() is NOT called when validation fails.
TEST_F(EpCompatibilityTest, TestEarlyValidation_FailsBeforeGetCapability) {
  const std::string ep_type = TestEarlyValidationExecutionProvider::kTestEarlyValidationExecutionProviderType;
  const std::string compatibility_string = "test_compatibility_v1.0";

  auto test_ep = std::make_unique<TestEarlyValidationExecutionProvider>();
  test_ep->SetMockCompatibilityStatus(OrtCompiledModelCompatibility_EP_UNSUPPORTED);

  // Verify GetCapability hasn't been called yet
  EXPECT_FALSE(test_ep->WasGetCapabilityCalled());

  // Create model with compatibility metadata for this EP
  std::map<std::string, std::string> compatibility_info = {{ep_type, compatibility_string}};
  auto model_with_metadata = ModelBuilderWithCompatibility::CreateModelWithCompatibilityMetadata(compatibility_info);

  auto session = SessionBuilderWithCompatibility::CreateTestSession(std::move(model_with_metadata));

  // Keep a raw pointer to check state after move
  auto* test_ep_ptr = test_ep.get();

  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::move(test_ep)));

  // Initialization should fail due to incompatible model
  auto status = InitializeSessionWithValidation(*session);
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("not supported"));

  // CRITICAL: GetCapability should NOT have been called because validation failed early,
  // before Initialize() could perform graph partitioning
  EXPECT_FALSE(test_ep_ptr->WasGetCapabilityCalled())
      << "GetCapability was called, indicating validation did not fail early before Initialize()";
}

// Test that when validation succeeds, GetCapability IS called (normal flow)
TEST_F(EpCompatibilityTest, TestEarlyValidation_SucceedsAndProceedsToGetCapability) {
  const std::string ep_type = TestEarlyValidationExecutionProvider::kTestEarlyValidationExecutionProviderType;
  const std::string compatibility_string = "test_compatibility_v1.0";

  auto test_ep = std::make_unique<TestEarlyValidationExecutionProvider>();
  test_ep->SetMockCompatibilityStatus(OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL);

  // Verify GetCapability hasn't been called yet
  EXPECT_FALSE(test_ep->WasGetCapabilityCalled());

  // Create model with compatibility metadata for this EP
  std::map<std::string, std::string> compatibility_info = {{ep_type, compatibility_string}};
  auto model_with_metadata = ModelBuilderWithCompatibility::CreateModelWithCompatibilityMetadata(compatibility_info);

  auto session = SessionBuilderWithCompatibility::CreateTestSession(std::move(model_with_metadata));

  // Keep a raw pointer to check state after move
  auto* test_ep_ptr = test_ep.get();

  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::move(test_ep)));

  // Initialization should succeed
  ASSERT_STATUS_OK(InitializeSessionWithValidation(*session));

  // GetCapability SHOULD have been called because validation succeeded and
  // Initialize() proceeded normally with graph partitioning
  EXPECT_TRUE(test_ep_ptr->WasGetCapabilityCalled())
      << "GetCapability was not called, but it should have been after successful validation";
}

// Test session option configuration for fail on suboptimal
TEST_F(EpCompatibilityTest, TestSessionOptionConfiguration) {
  SessionOptions so;

  // Test default value
  std::string config_value;
  bool has_config = so.config_options.TryGetConfigEntry(kOrtSessionOptionsFailOnSuboptimalCompiledModel, config_value);
  EXPECT_FALSE(has_config);  // Should not be set by default

  // Test setting the option
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsFailOnSuboptimalCompiledModel, "1"));
  has_config = so.config_options.TryGetConfigEntry(kOrtSessionOptionsFailOnSuboptimalCompiledModel, config_value);
  EXPECT_TRUE(has_config);
  EXPECT_EQ(config_value, "1");

  // Test setting to disabled
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsFailOnSuboptimalCompiledModel, "0"));
  has_config = so.config_options.TryGetConfigEntry(kOrtSessionOptionsFailOnSuboptimalCompiledModel, config_value);
  EXPECT_TRUE(has_config);
  EXPECT_EQ(config_value, "0");
}

// -----------------------------
// C API unit tests
// -----------------------------

namespace {

// Helper to create an OrtEnv and fetch a CPU EP device pointer via the C API.
// Returns a pair of (env, cpu_device). Caller releases env via api->ReleaseEnv.
static std::pair<OrtEnv*, const OrtEpDevice*> CreateEnvAndGetCpuEpDevice(const OrtApi* api) {
  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpCompatCapiTest", &env));
  EXPECT_NE(env, nullptr);

  const OrtEpDevice* const* devices = nullptr;
  size_t num_devices = 0;
  EXPECT_EQ(nullptr, api->GetEpDevices(env, &devices, &num_devices));
  EXPECT_GT(num_devices, 0u);

  const OrtEpDevice* cpu_device = nullptr;
  for (size_t i = 0; i < num_devices; ++i) {
    const char* name = api->EpDevice_EpName(devices[i]);
    if (name && std::string(name) == "CPUExecutionProvider") {
      cpu_device = devices[i];
      break;
    }
  }

  // Fallback: just pick the first device if CPU wasn't found (environment-dependent builds).
  if (!cpu_device && num_devices > 0) {
    cpu_device = devices[0];
  }

  EXPECT_NE(cpu_device, nullptr);
  return {env, cpu_device};
}

}  // namespace

TEST(EpCompatibilityCapiTest, InvalidArguments) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtCompiledModelCompatibility out_status = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;

  // ep_devices == nullptr
  OrtStatus* st = api->GetModelCompatibilityForEpDevices(nullptr, 0, "info", &out_status);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // Prepare a valid device
  auto [env, device] = CreateEnvAndGetCpuEpDevice(api);
  ASSERT_NE(env, nullptr);
  ASSERT_NE(device, nullptr);

  // compatibility_info == nullptr
  const OrtEpDevice* devices1[] = {device};
  st = api->GetModelCompatibilityForEpDevices(devices1, 1, nullptr, &out_status);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // out_status == nullptr
  st = api->GetModelCompatibilityForEpDevices(devices1, 1, "some-info", nullptr);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(EpCompatibilityCapiTest, CpuEpReturnsNotApplicableIfNoValidation) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  auto [env, device] = CreateEnvAndGetCpuEpDevice(api);
  ASSERT_NE(env, nullptr);
  ASSERT_NE(device, nullptr);

  OrtCompiledModelCompatibility out_status = static_cast<OrtCompiledModelCompatibility>(-1);
  const OrtEpDevice* devices2[] = {device};
  OrtStatus* st = api->GetModelCompatibilityForEpDevices(devices2, 1, "arbitrary-compat-string", &out_status);
  ASSERT_EQ(st, nullptr) << (st ? api->GetErrorMessage(st) : "");

  // For providers that don't implement validation, API should return EP_NOT_APPLICABLE.
  EXPECT_EQ(out_status, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

// -----------------------------
// C++ API unit tests
// -----------------------------

TEST(EpCompatibilityCxxApiTest, SingleDeviceCpuProvider) {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "EpCompatCxx"};
  auto devices = env.GetEpDevices();
  ASSERT_FALSE(devices.empty());

  std::vector<Ort::ConstEpDevice> selected;
  for (const auto& d : devices) {
    if (std::string{d.EpName()} == "CPUExecutionProvider") {
      selected.push_back(d);
      break;
    }
  }

  ASSERT_FALSE(selected.empty());

  // Pick a status that the CPU EP would never return to ensure the value is set correctly.
  OrtCompiledModelCompatibility status = OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION;
  ASSERT_NO_FATAL_FAILURE({
    status = Ort::GetModelCompatibilityForEpDevices(selected, "arbitrary-compat-string");
  });

  ASSERT_TRUE(status == OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);
}

// -----------------------------
// GetCompatibilityInfoFromModel Tests
// -----------------------------

TEST(EpCompatibilityCapiTest, GetCompatibilityInfoFromModel_InvalidArgs) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtAllocator* allocator = nullptr;
  ASSERT_EQ(api->GetAllocatorWithDefaultOptions(&allocator), nullptr);
  ASSERT_NE(allocator, nullptr);

  char* compat_info = nullptr;

  // model_path == nullptr
  OrtStatus* st = api->GetCompatibilityInfoFromModel(nullptr, "TestEP", allocator, &compat_info);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // ep_type == nullptr
  st = api->GetCompatibilityInfoFromModel(ORT_TSTR("test.onnx"), nullptr, allocator, &compat_info);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // ep_type == empty string
  st = api->GetCompatibilityInfoFromModel(ORT_TSTR("test.onnx"), "", allocator, &compat_info);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // allocator == nullptr
  st = api->GetCompatibilityInfoFromModel(ORT_TSTR("test.onnx"), "TestEP", nullptr, &compat_info);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // compatibility_info == nullptr
  st = api->GetCompatibilityInfoFromModel(ORT_TSTR("test.onnx"), "TestEP", allocator, nullptr);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);
}

TEST(EpCompatibilityCapiTest, GetCompatibilityInfoFromModel_FileNotFound) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtAllocator* allocator = nullptr;
  ASSERT_EQ(api->GetAllocatorWithDefaultOptions(&allocator), nullptr);
  ASSERT_NE(allocator, nullptr);

  char* compat_info = nullptr;
  OrtStatus* st = api->GetCompatibilityInfoFromModel(ORT_TSTR("nonexistent_model.onnx"), "TestEP", allocator, &compat_info);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_NO_SUCHFILE);
  api->ReleaseStatus(st);
}

TEST(EpCompatibilityCapiTest, GetCompatibilityInfoFromModelBytes_InvalidArgs) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtAllocator* allocator = nullptr;
  ASSERT_EQ(api->GetAllocatorWithDefaultOptions(&allocator), nullptr);
  ASSERT_NE(allocator, nullptr);

  char* compat_info = nullptr;
  const char dummy_data[] = "dummy";

  // model_data == nullptr
  OrtStatus* st = api->GetCompatibilityInfoFromModelBytes(nullptr, 10, "TestEP", allocator, &compat_info);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // model_data_length == 0
  st = api->GetCompatibilityInfoFromModelBytes(dummy_data, 0, "TestEP", allocator, &compat_info);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // ep_type == nullptr
  st = api->GetCompatibilityInfoFromModelBytes(dummy_data, sizeof(dummy_data), nullptr, allocator, &compat_info);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // ep_type == empty string
  st = api->GetCompatibilityInfoFromModelBytes(dummy_data, sizeof(dummy_data), "", allocator, &compat_info);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // allocator == nullptr
  st = api->GetCompatibilityInfoFromModelBytes(dummy_data, sizeof(dummy_data), "TestEP", nullptr, &compat_info);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // compatibility_info == nullptr
  st = api->GetCompatibilityInfoFromModelBytes(dummy_data, sizeof(dummy_data), "TestEP", allocator, nullptr);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // model_data_length > INT_MAX (should return error, not crash)
  // We can't actually allocate this much memory, but we can pass the size
  // The API should validate the size before attempting to use the data
  size_t oversized_length = static_cast<size_t>(INT_MAX) + 1;
  st = api->GetCompatibilityInfoFromModelBytes(dummy_data, oversized_length, "TestEP", allocator, &compat_info);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);
}

TEST(EpCompatibilityCapiTest, GetCompatibilityInfoFromModelBytes_InvalidModelData) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtAllocator* allocator = nullptr;
  ASSERT_EQ(api->GetAllocatorWithDefaultOptions(&allocator), nullptr);
  ASSERT_NE(allocator, nullptr);

  char* compat_info = nullptr;
  const char invalid_data[] = "this is not a valid ONNX model";

  OrtStatus* st = api->GetCompatibilityInfoFromModelBytes(invalid_data, sizeof(invalid_data), "TestEP", allocator, &compat_info);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_GRAPH);
  api->ReleaseStatus(st);
}

// Test extracting compatibility info from a model with metadata
TEST(EpCompatibilityCapiTest, GetCompatibilityInfoFromModelBytes_WithMetadata) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtAllocator* allocator = nullptr;
  ASSERT_EQ(api->GetAllocatorWithDefaultOptions(&allocator), nullptr);
  ASSERT_NE(allocator, nullptr);

  // Create a minimal ModelProto with compatibility metadata
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model_proto.mutable_graph()->set_name("test_graph");

  // Add an opset import (required)
  auto* opset = model_proto.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  // Add compatibility metadata
  const std::string ep_type = "TestCompatEP";
  const std::string expected_compat_info = "test_compat_v1.0_driver_123";
  auto* prop = model_proto.add_metadata_props();
  prop->set_key(std::string("ep_compatibility_info.") + ep_type);
  prop->set_value(expected_compat_info);

  // Serialize the model
  std::string model_data;
  ASSERT_TRUE(model_proto.SerializeToString(&model_data));

  // Extract compatibility info
  char* compat_info = nullptr;
  OrtStatus* st = api->GetCompatibilityInfoFromModelBytes(
      model_data.data(), model_data.size(), ep_type.c_str(), allocator, &compat_info);
  ASSERT_EQ(st, nullptr) << (st ? api->GetErrorMessage(st) : "");
  ASSERT_NE(compat_info, nullptr);
  EXPECT_STREQ(compat_info, expected_compat_info.c_str());
  ASSERT_EQ(api->AllocatorFree(allocator, compat_info), nullptr);
}

// Test when compatibility info is not found for the EP
TEST(EpCompatibilityCapiTest, GetCompatibilityInfoFromModelBytes_NotFound) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtAllocator* allocator = nullptr;
  ASSERT_EQ(api->GetAllocatorWithDefaultOptions(&allocator), nullptr);
  ASSERT_NE(allocator, nullptr);

  // Create a minimal ModelProto without compatibility metadata for our EP
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model_proto.mutable_graph()->set_name("test_graph");

  auto* opset = model_proto.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  // Add metadata for a different EP
  auto* prop = model_proto.add_metadata_props();
  prop->set_key("ep_compatibility_info.DifferentEP");
  prop->set_value("some_value");

  std::string model_data;
  ASSERT_TRUE(model_proto.SerializeToString(&model_data));

  // Try to get compatibility info for an EP that doesn't have it
  char* compat_info = nullptr;
  OrtStatus* st = api->GetCompatibilityInfoFromModelBytes(
      model_data.data(), model_data.size(), "NonExistentEP", allocator, &compat_info);
  ASSERT_EQ(st, nullptr);           // Not an error - just not found
  EXPECT_EQ(compat_info, nullptr);  // Should be nullptr when not found
}

// C++ API test
TEST(EpCompatibilityCxxApiTest, GetCompatibilityInfoFromModelBytes) {
  // Create a minimal ModelProto with compatibility metadata
  ONNX_NAMESPACE::ModelProto model_proto;
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model_proto.mutable_graph()->set_name("test_graph");

  auto* opset = model_proto.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  const std::string ep_type = "CxxTestEP";
  const std::string expected_compat_info = "cxx_compat_v2.0";
  auto* prop = model_proto.add_metadata_props();
  prop->set_key(std::string("ep_compatibility_info.") + ep_type);
  prop->set_value(expected_compat_info);

  std::string model_data;
  ASSERT_TRUE(model_proto.SerializeToString(&model_data));

  // Get allocator
  Ort::AllocatorWithDefaultOptions allocator;

  // Test C++ API - found case
  Ort::AllocatedStringPtr result = Ort::GetCompatibilityInfoFromModelBytesAllocated(
      model_data.data(), model_data.size(), ep_type.c_str(), allocator);
  ASSERT_NE(result.get(), nullptr);
  EXPECT_STREQ(result.get(), expected_compat_info.c_str());

  // Test when not found - should return nullptr
  Ort::AllocatedStringPtr not_found = Ort::GetCompatibilityInfoFromModelBytesAllocated(
      model_data.data(), model_data.size(), "NonExistentEP", allocator);
  EXPECT_EQ(not_found.get(), nullptr);
}
