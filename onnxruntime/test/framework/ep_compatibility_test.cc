// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
#include "core/session/abi_session_options_impl.h"
#include "core/framework/error_code_helper.h"
#include "dummy_provider.h"
#include "test_utils.h"
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

  // ep_device == nullptr
  OrtStatus* st = api->GetEpCompatibilityForDevice(nullptr, "info", &out_status);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // Prepare a valid device
  auto [env, device] = CreateEnvAndGetCpuEpDevice(api);
  ASSERT_NE(env, nullptr);
  ASSERT_NE(device, nullptr);

  // compatibility_info == nullptr
  st = api->GetEpCompatibilityForDevice(device, nullptr, &out_status);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  // out_status == nullptr
  st = api->GetEpCompatibilityForDevice(device, "some-info", nullptr);
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
  OrtStatus* st = api->GetEpCompatibilityForDevice(device, "arbitrary-compat-string", &out_status);
  ASSERT_EQ(st, nullptr) << (st ? api->GetErrorMessage(st) : "");

  // For providers that don't implement validation, API should return EP_NOT_APPLICABLE.
  EXPECT_EQ(out_status, OrtCompiledModelCompatibility_EP_NOT_APPLICABLE);

  api->ReleaseEnv(env);
}
