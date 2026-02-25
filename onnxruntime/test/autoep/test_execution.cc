// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <vector>
// #include <absl/base/config.h>
#include <gsl/gsl>
#include <gtest/gtest.h>

#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"

#include "test/autoep/test_autoep_utils.h"
#include "test/shared_lib/utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

namespace {

void RunMulModelWithPluginEp(const Ort::SessionOptions& session_options) {
  Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);

  // Create input
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<int64_t> shape = {3, 2};
  std::vector<float> input0_data(6, 2.0f);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> ort_input_names;

  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, input0_data.data(), input0_data.size(), shape.data(), shape.size()));
  ort_input_names.push_back("X");

  // Run session and get outputs
  std::array<const char*, 1> output_names{"Y"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check expected output values
  Ort::Value& ort_output = ort_outputs[0];
  const float* output_data = ort_output.GetTensorData<float>();
  gsl::span<const float> output_span(output_data, 6);
  EXPECT_THAT(output_span, ::testing::ElementsAre(2, 4, 6, 8, 10, 12));
}

void RunCustomMulModelWithPluginEp(const Ort::SessionOptions& session_options) {
  Ort::Session session(*ort_env, ORT_TSTR("testdata/custom_mul.onnx"), session_options);

  // Create two inputs with same values
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<int64_t> shape = {3, 2};
  std::vector<float> input0_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> ort_input_names;

  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, input0_data.data(), input0_data.size(), shape.data(), shape.size()));
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, input0_data.data(), input0_data.size(), shape.data(), shape.size()));
  ort_input_names.push_back("X");
  ort_input_names.push_back("W");

  // Run session and get outputs
  std::array<const char*, 1> output_names{"Y"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check expected output values
  Ort::Value& ort_output = ort_outputs[0];
  const float* output_data = ort_output.GetTensorData<float>();
  gsl::span<const float> output_span(output_data, 6);
  EXPECT_THAT(output_span, ::testing::ElementsAre(1, 4, 9, 16, 25, 36));
}

void RunSqueezeMulReluModel(const Ort::SessionOptions& session_options) {
  Ort::Session session(*ort_env, ORT_TSTR("testdata/squeeze_mul_relu.onnx"), session_options);

  // Create inputs
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<int64_t, 3> a_shape = {3, 1, 2};
  std::array<int64_t, 2> b_shape = {3, 2};

  std::array<float, 6> a_data = {1.f, -2.f, 3.f, 4.f, -5.f, 6.f};
  std::array<float, 6> b_data = {2.f, 3.f, 4.f, -5.f, 6.f, 7.f};

  std::vector<Ort::Value> ort_inputs{};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(memory_info, a_data.data(), a_data.size(), a_shape.data(), a_shape.size()));
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(memory_info, b_data.data(), b_data.size(), b_shape.data(), b_shape.size()));

  std::array ort_input_names{"A", "B"};

  // Run session and get outputs
  std::array output_names{"C"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check expected output values
  Ort::Value& ort_output = ort_outputs[0];
  const float* output_data = ort_output.GetTensorData<float>();
  gsl::span<const float> output_span(output_data, 6);
  EXPECT_THAT(output_span, ::testing::ElementsAre(4, 0, 24, 0, 0, 84));
}

void RunSubMulSubModel(const Ort::SessionOptions& session_options) {
  // This model has Sub -> Mul -> Sub: (A - B) * B - A
  // The example plugin EP supports all ops.
  Ort::Session session(*ort_env, ORT_TSTR("testdata/sub_mul_sub.onnx"), session_options);

  // Create inputs
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<int64_t> shape = {3, 2};

  std::vector<float> a_data{1, 2, 3, 4, 5, 6};
  std::vector<float> b_data{2, 3, 4, 5, 6, 7};

  std::vector<Ort::Value> ort_inputs{};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(memory_info, a_data.data(), a_data.size(), shape.data(), shape.size()));
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(memory_info, b_data.data(), b_data.size(), shape.data(), shape.size()));

  std::array ort_input_names{"A", "B"};

  // Run session and get outputs
  std::array output_names{"C"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check expected output values
  Ort::Value& ort_output = ort_outputs[0];
  const float* output_data = ort_output.GetTensorData<float>();
  gsl::span<const float> output_span(output_data, 6);
  EXPECT_THAT(output_span, ::testing::ElementsAre(-3, -5, -7, -9, -11, -13));
}

void RunIfMulModel(const Ort::SessionOptions& session_options, bool if_condition) {
  // Model graph does the following computation:
  // if (A) { C = B * 2.0; }
  // else { C = B * 3; }
  Ort::Session session(*ort_env, ORT_TSTR("testdata/if_mul.onnx"), session_options);

  // Create inputs
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<int64_t, 1> a_shape = {1};
  std::array<int64_t, 2> b_shape = {3, 2};

  std::array<bool, 1> a_data = {if_condition};
  std::array<float, 6> b_data = {2.f, 3.f, 4.f, -5.f, 6.f, 7.f};

  std::vector<Ort::Value> ort_inputs{};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<bool>(memory_info, a_data.data(), a_data.size(), a_shape.data(), a_shape.size()));
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(memory_info, b_data.data(), b_data.size(), b_shape.data(), b_shape.size()));

  std::array ort_input_names{"A", "B"};

  // Run session and get outputs
  std::array output_names{"C"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check expected output values
  Ort::Value& ort_output = ort_outputs[0];
  const float* output_data = ort_output.GetTensorData<float>();
  gsl::span<const float> output_span(output_data, 6);

  if (if_condition) {
    // Expect that model multiplies input B elements by 2.0.
    EXPECT_THAT(output_span, ::testing::ElementsAre(4.f, 6.f, 8.f, -10.f, 12.f, 14.f));
  } else {
    // Expect that model multiplies input B elements by 3.0.
    EXPECT_THAT(output_span, ::testing::ElementsAre(6.f, 9.f, 12.f, -15.f, 18.f, 21.f));
  }
}

void RunLoopSubOneModel(const Ort::SessionOptions& session_options) {
  // Model graph does the following computation:
  // x = A
  // for (int i = 0; i < MAX_ITERS; i++) {
  //   y = x - 1.0;
  //   user_val = x - 1.0;
  //   x = y;
  // }
  // C = x;
  // D = user_val (will be concatenated result of each iteration)
  Ort::Session session(*ort_env, ORT_TSTR("testdata/loop_sub_one.onnx"), session_options);

  // Create inputs
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<int64_t, 1> max_iters_shape = {1};
  std::array<int64_t, 1> a_shape = {1};

  std::array<int64_t, 1> max_iters_data = {3};
  std::array<float, 1> a_data = {10.0f};

  std::vector<Ort::Value> ort_inputs{};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<int64_t>(memory_info, max_iters_data.data(), max_iters_data.size(),
                                        max_iters_shape.data(), max_iters_shape.size()));
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(memory_info, a_data.data(), a_data.size(), a_shape.data(), a_shape.size()));

  std::array ort_input_names{"MAX_ITERS", "A"};

  // Run session and get outputs
  std::array output_names{"C", "D"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check expected output values
  // Expect that input elems are all subtracted by 3 (1 each iteration).
  gsl::span<const float> output_c_span(ort_outputs[0].GetTensorData<float>(), 1);
  EXPECT_THAT(output_c_span, ::testing::ElementsAre(7.f));

  gsl::span<const float> output_d_span(ort_outputs[1].GetTensorData<float>(), 3);
  EXPECT_THAT(output_d_span, ::testing::ElementsAre(9.f, 8.f, 7.f));
}

void RunScanMulModel(const Ort::SessionOptions& session_options) {
  Ort::Session session(*ort_env, ORT_TSTR("testdata/scan_mul.onnx"), session_options);

  // Create inputs
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<int64_t, 2> x_shape = {3, 3};
  std::array<float, 9> x_data = {1.f, 2.f, 3.f, 10.f, 20.f, 30.f, 100.f, 200.f, 300.f};

  std::vector<Ort::Value> ort_inputs{};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(memory_info, x_data.data(), x_data.size(), x_shape.data(), x_shape.size()));

  std::array ort_input_names{"X"};

  // Run session and get outputs
  std::array output_names{"Y"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check expected output values
  gsl::span<const float> output_span(ort_outputs[0].GetTensorData<float>(), 9);
  EXPECT_THAT(output_span, ::testing::ElementsAre(2.f, 4.f, 6.f, 20.f, 40.f, 60.f, 200.f, 400.f, 600.f));
}

using CheckEpNodeAssignmentFunc = std::function<void(const Ort::Session& session)>;

void RunAddMulAddModel(const Ort::SessionOptions& session_options,
                       CheckEpNodeAssignmentFunc check_ep_node_assignment_func = {}) {
  // This model has Add -> Mul -> Add. The example plugin EP supports Mul but not Add.
  Ort::Session session(*ort_env, ORT_TSTR("testdata/add_mul_add.onnx"), session_options);

  if (check_ep_node_assignment_func) {
    ASSERT_NO_FATAL_FAILURE(check_ep_node_assignment_func(session));
  }

  // Create inputs
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<int64_t> shape = {3, 2};

  std::vector<float> a_data{1, 2, 3, 4, 5, 6};
  std::vector<float> b_data{2, 3, 4, 5, 6, 7};

  std::vector<Ort::Value> ort_inputs{};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(memory_info, a_data.data(), a_data.size(), shape.data(), shape.size()));
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(memory_info, b_data.data(), b_data.size(), shape.data(), shape.size()));

  std::array ort_input_names{"A", "B"};

  // Run session and get outputs
  std::array output_names{"C"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check expected output values
  Ort::Value& ort_output = ort_outputs[0];
  const float* output_data = ort_output.GetTensorData<float>();
  gsl::span<const float> output_span(output_data, 6);
  EXPECT_THAT(output_span, ::testing::ElementsAre(7, 17, 31, 49, 71, 97));
}

}  // namespace

// Creates a session with the example plugin EP and runs a model with a single Mul node.
// Uses AppendExecutionProvider_V2 to append the example plugin EP to the session.
TEST(OrtEpLibrary, PluginEp_AppendV2_MulInference) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  // Create session with example plugin EP
  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  RunMulModelWithPluginEp(session_options);
}

// Creates a session with the example plugin EP and runs a model with a single Mul node.
// Uses the PREFER_CPU policy to append the example plugin EP to the session.
TEST(OrtEpLibrary, PluginEp_PreferCpu_MulInference) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));

  {
    // PREFER_CPU pick our example EP over ORT CPU EP. TODO: Actually assert this.
    Ort::SessionOptions session_options;
    session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_CPU);
    RunMulModelWithPluginEp(session_options);
  }
}

TEST(OrtEpLibrary, PluginEp_AppendV2_PartiallySupportedModelInference) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  // Create session with example plugin EP
  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;

  // Create session that enables recording of EP-graph assignment info
  session_options.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  // Function that checks the ep graph/node assignment (Mul on plugin EP, others on CPU EP).
  // Model has 3 subgraphs (in no particular order):
  // - Subgraph 1: Add assigned to CPU EP.
  // - Subgraph 2: Mul assigned to plugin EP.
  // - Subgraph 3: Add assigned to CPU EP.
  auto check_ep_node_assignment = [](const Ort::Session& session) -> void {
    std::vector<Ort::ConstEpAssignedSubgraph> ep_subgraphs = session.GetEpGraphAssignmentInfo();
    ASSERT_EQ(ep_subgraphs.size(), 3);

    for (Ort::ConstEpAssignedSubgraph ep_subgraph : ep_subgraphs) {
      std::string ep_name = ep_subgraph.GetEpName();
      ASSERT_TRUE(ep_name == Utils::example_ep_info.ep_name || ep_name == kCpuExecutionProvider);

      const std::vector<Ort::ConstEpAssignedNode> ep_nodes = ep_subgraph.GetNodes();

      ASSERT_GE(ep_nodes.size(), 1);  // All of these subgraphs just have one node.
      std::string domain = ep_nodes[0].GetDomain();
      std::string op_type = ep_nodes[0].GetOperatorType();
      std::string node_name = ep_nodes[0].GetName();

      ASSERT_EQ(domain, kOnnxDomain);  // All node ops should have the ONNX domain

      // Check that CPU EP has the Adds and that the example EP has the Mul.
      if (ep_name == kCpuExecutionProvider) {
        ASSERT_EQ(op_type, "Add");
        ASSERT_TRUE(node_name == "add_0" || node_name == "add_1");
      } else {
        ASSERT_TRUE(ep_name == Utils::example_ep_info.ep_name);
        ASSERT_EQ(op_type, "Mul");
        ASSERT_EQ(node_name, "mul_0");
      }
    }
  };

  RunAddMulAddModel(session_options, check_ep_node_assignment);
}

// Generate an EPContext model with a plugin EP.
// This test uses the OrtCompileApi but could also be done by setting the appropriate session option configs.
TEST(OrtEpLibrary, PluginEp_GenEpContextModel) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  {
    const ORTCHAR_T* input_model_file = ORT_TSTR("testdata/mul_1.onnx");
    const ORTCHAR_T* output_model_file = ORT_TSTR("plugin_ep_mul_1_ctx.onnx");
    std::filesystem::remove(output_model_file);

    // Create session with example plugin EP
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;

    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    // Create model compilation options from the session options.
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(output_model_file);

    // Compile the model.
    ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
    // Make sure the compiled model was generated.
    ASSERT_TRUE(std::filesystem::exists(output_model_file));
  }
}

// Generate an EPContext model with a plugin EP that uses a virtual GPU.
TEST(OrtEpLibrary, PluginEp_VirtGpu_GenEpContextModel) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_virt_gpu_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  {
    const ORTCHAR_T* input_model_file = ORT_TSTR("testdata/add_mul_add.onnx");
    const ORTCHAR_T* output_model_file = ORT_TSTR("plugin_ep_virt_gpu_add_mul_add_ctx.onnx");
    std::filesystem::remove(output_model_file);

    // Create session with example plugin EP
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;

    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    // Create model compilation options from the session options.
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(output_model_file);

    // Compile the model.
    ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
    // Make sure the compiled model was generated.
    ASSERT_TRUE(std::filesystem::exists(output_model_file));
  }
}

// Test that compatibility info is written to compiled model metadata
TEST(OrtEpLibrary, PluginEp_CompatibilityInfo_WrittenToMetadata) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  const ORTCHAR_T* input_model_file = ORT_TSTR("testdata/mul_1.onnx");
  const ORTCHAR_T* output_model_file = ORT_TSTR("plugin_ep_compat_test.onnx");
  std::filesystem::remove(output_model_file);

  // Compile the model
  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(output_model_file);

    ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
    ASSERT_TRUE(std::filesystem::exists(output_model_file));
  }

  // Load the compiled model and check metadata for compatibility info
  {
    Ort::SessionOptions session_options;
    // Need to add the EP to handle EPContext nodes
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    Ort::Session session(*ort_env, output_model_file, session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // Check that the model has EP compatibility metadata
    Ort::ModelMetadata metadata = session.GetModelMetadata();
    auto custom_metadata_keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);

    // Check for the exact metadata key for this EP: "ep_compatibility_info.example_ep"
    const std::string expected_key = std::string(kOrtModelMetadata_EpCompatibilityInfoPrefix) + "example_ep";

    bool found_compatibility_key = false;
    for (const auto& key : custom_metadata_keys) {
      std::string key_str(key.get());
      if (key_str == expected_key) {
        found_compatibility_key = true;
        break;
      }
    }
    ASSERT_TRUE(found_compatibility_key) << "Expected metadata key '" << expected_key << "' in compiled model";

    // Verify the compatibility value contains expected EP information
    auto value = metadata.LookupCustomMetadataMapAllocated(expected_key.c_str(), allocator);
    ASSERT_NE(value.get(), nullptr);
    std::string compatibility_value = value.get();
    ASSERT_GT(compatibility_value.length(), 0) << "Compatibility info should not be empty";

    // Validate the exact compatibility string format and values
    // Format: "example_ep;version=0.1.0;ort_api_version=<ORT_API_VERSION>"
    std::string expected_compatibility_info = "example_ep;version=0.1.0;ort_api_version=" +
                                              std::to_string(ORT_API_VERSION);
    EXPECT_EQ(compatibility_value, expected_compatibility_info);
  }

  std::filesystem::remove(output_model_file);
}

// Test loading a compiled model validates compatibility successfully
TEST(OrtEpLibrary, PluginEp_CompatibilityInfo_ValidatedOnLoad) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  const ORTCHAR_T* input_model_file = ORT_TSTR("testdata/mul_1.onnx");
  const ORTCHAR_T* compiled_model_file = ORT_TSTR("plugin_ep_compat_validate.onnx");
  std::filesystem::remove(compiled_model_file);

  // Step 1: Compile the model
  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(compiled_model_file);

    ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
    ASSERT_TRUE(std::filesystem::exists(compiled_model_file));
  }

  // Step 2: Load the compiled model with the same EP - should succeed
  // The EP should validate compatibility and return OPTIMAL status
  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    // This should not throw - EP should validate compatibility as OPTIMAL
    ASSERT_NO_THROW(Ort::Session session(*ort_env, compiled_model_file, session_options));
  }

  std::filesystem::remove(compiled_model_file);
}

// Test that loading a compiled model with ep_context_enable=1 returns an error.
// This is an invalid configuration: the user is asking to generate EP context from a model
// that already contains EPContext nodes.
TEST(OrtEpLibrary, PluginEp_Error_LoadCompiledModelWithEpContextEnabled) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  const ORTCHAR_T* input_model_file = ORT_TSTR("testdata/mul_1.onnx");
  const ORTCHAR_T* compiled_model_file = ORT_TSTR("plugin_ep_recompile_test.onnx");
  std::filesystem::remove(compiled_model_file);

  // Step 1: Compile the original model (CompileModel API implicitly generates EPContext nodes)
  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(compiled_model_file);

    ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
    ASSERT_TRUE(std::filesystem::exists(compiled_model_file));
  }

  // Step 2: Attempt to load the compiled model with ep.context_enable=1 - should fail
  {
    Ort::SessionOptions session_options;
    session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");  // Request EP context generation
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    // Loading a compiled model with ep_context_enable=1 should fail
    try {
      Ort::Session session(*ort_env, compiled_model_file, session_options);
      FAIL() << "Expected error when loading compiled model with ep_context_enable=1";
    } catch (const Ort::Exception& e) {
      std::string error_msg = e.what();
      // Verify error message mentions the issue
      EXPECT_TRUE(error_msg.find("EPContext") != std::string::npos ||
                  error_msg.find("already") != std::string::npos ||
                  error_msg.find("re-compile") != std::string::npos)
          << "Error should mention EPContext or re-compilation: " << error_msg;
    }
  }

  std::filesystem::remove(compiled_model_file);
}

// Test that EPContext inference returns expected "not implemented" error.
// This documents that the example EP does not fully support EPContext execution.
TEST(OrtEpLibrary, PluginEp_EpContextInference_NotImplemented) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  const ORTCHAR_T* input_model_file = ORT_TSTR("testdata/mul_1.onnx");
  const ORTCHAR_T* compiled_model_file = ORT_TSTR("plugin_ep_inference_test.onnx");
  std::filesystem::remove(compiled_model_file);

  // Step 1: Compile the model with EP context enabled
  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetFlags(OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(compiled_model_file);

    ASSERT_CXX_ORTSTATUS_OK(Ort::CompileModel(*ort_env, compile_options));
    ASSERT_TRUE(std::filesystem::exists(compiled_model_file));
  }

  // Step 2: Load compiled model and attempt inference - should fail with clear error
  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    Ort::Session session(*ort_env, compiled_model_file, session_options);

    // Prepare inputs - mul_1.onnx has input X of shape [3,2]
    std::vector<float> input_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<int64_t> input_shape = {3, 2};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value input_x_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_x.data(), input_x.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_x_tensor));

    // Inference should fail with NOT_IMPLEMENTED - verify exception content
    try {
      auto outputs = session.Run(Ort::RunOptions{nullptr},
                                 input_names, input_tensors.data(), input_tensors.size(),
                                 output_names, 1);
      FAIL() << "Expected exception for EPContext inference, but Run() succeeded";
    } catch (const Ort::Exception& e) {
      std::string msg = e.what();
      // Verify error mentions the limitation
      EXPECT_TRUE(msg.find("not implemented") != std::string::npos ||
                  msg.find("NOT_IMPLEMENTED") != std::string::npos ||
                  msg.find("EPContext") != std::string::npos)
          << "Expected NOT_IMPLEMENTED or EPContext in error, got: " << msg;
    }
  }

  std::filesystem::remove(compiled_model_file);
}

// Uses the original compiling approach with session option configs (instead of explicit compile API).
// Test that ORT does not overwrite an output model if it already exists.
// Notably, this tests the case in which ORT automatically generates the output model name.
TEST(OrtEpLibrary, PluginEp_GenEpContextModel_ErrorOutputModelExists_AutoGenOutputModelName) {
  const ORTCHAR_T* input_model_file = ORT_TSTR("testdata/mul_1.onnx");
  const ORTCHAR_T* expected_output_model_file = ORT_TSTR("testdata/mul_1_ctx.onnx");
  std::filesystem::remove(expected_output_model_file);

  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());
  std::unordered_map<std::string, std::string> ep_options;

  // Compile a model and let ORT set the output model name. This should succeed.
  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    // Don't specify an output model path to let ORT automatically generate it!
    // so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "");

    so.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    Ort::Session session(*ort_env, input_model_file, so);
    ASSERT_TRUE(std::filesystem::exists(expected_output_model_file));  // check compiled model was generated.
  }

  auto modify_time_1 = std::filesystem::last_write_time(expected_output_model_file);

  // Try compiling the model again. ORT should return an error because the output model already exists.
  // Original compiled model should not be modified.
  {
    Ort::SessionOptions so;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    // Don't specify an output model path to let ORT automatically generate it!
    // so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, "");

    so.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

    try {
      Ort::Session session(*ort_env, input_model_file, so);
      FAIL();  // Should not get here!
    } catch (const Ort::Exception& excpt) {
      ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_FAIL);
      ASSERT_THAT(excpt.what(),
                  testing::HasSubstr("exists already. "
                                     "Please remove the EP context model if you want to re-generate it."));

      ASSERT_TRUE(std::filesystem::exists(expected_output_model_file));
      auto modify_time_2 = std::filesystem::last_write_time(expected_output_model_file);
      ASSERT_EQ(modify_time_2, modify_time_1);  // Check that file was not modified
    }
  }

  std::filesystem::remove(expected_output_model_file);
}

TEST(OrtEpLibrary, KernelPluginEp_Inference) {
  RegisteredEpDeviceUniquePtr example_kernel_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_kernel_registry_info,
                                                         example_kernel_ep));
  Ort::ConstEpDevice plugin_ep_device(example_kernel_ep.get());

  // Run model with squeeze, mul, and relu nodes.
  // No sharing of pre-packed weights.
  {
    std::unordered_map<std::string, std::string> ep_options;
    Ort::SessionOptions session_options;

    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Fail if any node assigned to CPU EP.
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);
    ASSERT_NO_FATAL_FAILURE(RunSqueezeMulReluModel(session_options));
  }

  // Run model with squeeze, mul, and relu nodes.
  // Enable sharing of pre-packed weights.
  {
    std::unordered_map<std::string, std::string> ep_options = {{"enable_prepack_weight_sharing", "1"}};
    Ort::SessionOptions session_options;

    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Fail if any node assigned to CPU EP
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);
    ASSERT_NO_FATAL_FAILURE(RunSqueezeMulReluModel(session_options));
  }

  // Run model with sub, mul, sub.
  // No sharing of pre-packed weights.
  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;

    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Fail if any node assigned to CPU EP
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);
    ASSERT_NO_FATAL_FAILURE(RunSubMulSubModel(session_options));
  }

  // Run model with sub, mul, sub.
  // Enable sharing of pre-packed weights.
  {
    std::unordered_map<std::string, std::string> ep_options = {{"enable_prepack_weight_sharing", "1"}};
    Ort::SessionOptions session_options;

    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Fail if any node assigned to CPU EP
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);
    ASSERT_NO_FATAL_FAILURE(RunSubMulSubModel(session_options));
  }
}

TEST(OrtEpLibrary, KernelPluginEp_ControlFlow_If) {
  RegisteredEpDeviceUniquePtr example_kernel_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_kernel_registry_info,
                                                         example_kernel_ep));
  Ort::ConstEpDevice plugin_ep_device(example_kernel_ep.get());

  // Run model with If and Mul ops.
  // No sharing of pre-packed weights.
  {
    std::unordered_map<std::string, std::string> ep_options;
    Ort::SessionOptions session_options;

    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Fail if any node assigned to CPU EP.
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);
    ASSERT_NO_FATAL_FAILURE(RunIfMulModel(session_options, /*if_condition*/ true));
    ASSERT_NO_FATAL_FAILURE(RunIfMulModel(session_options, /*if_condition*/ false));
  }

  // Run model with If and Mul ops.
  // Enable sharing of pre-packed weights.
  {
    std::unordered_map<std::string, std::string> ep_options = {{"enable_prepack_weight_sharing", "1"}};
    Ort::SessionOptions session_options;

    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Fail if any node assigned to CPU EP
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);
    ASSERT_NO_FATAL_FAILURE(RunIfMulModel(session_options, /*if_condition*/ true));
    ASSERT_NO_FATAL_FAILURE(RunIfMulModel(session_options, /*if_condition*/ false));
  }
}

TEST(OrtEpLibrary, KernelPluginEp_ControlFlow_Loop) {
  RegisteredEpDeviceUniquePtr example_kernel_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_kernel_registry_info,
                                                         example_kernel_ep));
  Ort::ConstEpDevice plugin_ep_device(example_kernel_ep.get());

  // Run model with Loop and Sub ops.
  // No sharing of pre-packed weights.
  {
    std::unordered_map<std::string, std::string> ep_options;
    Ort::SessionOptions session_options;

    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Fail if any node assigned to CPU EP
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);
    ASSERT_NO_FATAL_FAILURE(RunLoopSubOneModel(session_options));
  }

  // Run model with Loop and Sub ops.
  // Enable sharing of pre-packed weights.
  {
    std::unordered_map<std::string, std::string> ep_options = {{"enable_prepack_weight_sharing", "1"}};
    Ort::SessionOptions session_options;

    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Fail if any node assigned to CPU EP
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);
    ASSERT_NO_FATAL_FAILURE(RunLoopSubOneModel(session_options));
  }
}

TEST(OrtEpLibrary, KernelPluginEp_ControlFlow_Scan) {
  RegisteredEpDeviceUniquePtr example_kernel_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_kernel_registry_info,
                                                         example_kernel_ep));
  Ort::ConstEpDevice plugin_ep_device(example_kernel_ep.get());

  // Run model with Scan and Mul ops.
  // No sharing of pre-packed weights.
  {
    std::unordered_map<std::string, std::string> ep_options;
    Ort::SessionOptions session_options;

    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Fail if any node assigned to CPU EP
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);
    ASSERT_NO_FATAL_FAILURE(RunScanMulModel(session_options));
  }

  // Run model with Scan and Mul ops.
  // Enable sharing of pre-packed weights.
  {
    std::unordered_map<std::string, std::string> ep_options = {{"enable_prepack_weight_sharing", "1"}};
    Ort::SessionOptions session_options;

    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Fail if any node assigned to CPU EP
    session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);
    ASSERT_NO_FATAL_FAILURE(RunScanMulModel(session_options));
  }
}

// Creates a session with the example plugin EP and runs a model with a single Costom_Mul node.
// Uses AppendExecutionProvider_V2 to append the example plugin EP to the session.
TEST(OrtEpLibrary, PluginEp_Custom_Op_Inference_With_Explicit_Ep) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  // Create session with example plugin EP
  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  RunCustomMulModelWithPluginEp(session_options);
}

// Creates a session with the example plugin EP and runs a model with a single Costom_Mul node.
// Uses the PREFER_CPU policy to append the example plugin EP to the session.
TEST(OrtEpLibrary, PluginEp_Custom_Op_Inference_With_Prefer_Cpu) {
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));
  Ort::ConstEpDevice plugin_ep_device(example_ep.get());

  {
    // PREFER_CPU pick our example EP over ORT CPU EP. TODO: Actually assert this.
    Ort::SessionOptions session_options;
    session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_CPU);
    RunCustomMulModelWithPluginEp(session_options);
  }
}

// Tests the GetHardwareDeviceEpIncompatibilityDetails C API with the example plugin EP.
// The example plugin EP supports CPU devices, so this test verifies that a CPU device
// is reported as compatible (reasons_bitmask == 0).
TEST(OrtEpLibrary, PluginEp_CpuDevice_ReturnsCompatible) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = static_cast<OrtEnv*>(*ort_env);

  // Register the example plugin EP
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));

  // Get all hardware devices
  size_t num_hw_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetNumHardwareDevices(env, &num_hw_devices));
  ASSERT_GT(num_hw_devices, 0u);
  std::vector<const OrtHardwareDevice*> hw_devices(num_hw_devices);
  ASSERT_ORTSTATUS_OK(api->GetHardwareDevices(env, hw_devices.data(), num_hw_devices));

  // Find a CPU device using the public accessor
  const OrtHardwareDevice* cpu_device = nullptr;
  for (size_t i = 0; i < num_hw_devices; ++i) {
    if (api->HardwareDevice_Type(hw_devices[i]) == OrtHardwareDeviceType_CPU) {
      cpu_device = hw_devices[i];
      break;
    }
  }
  ASSERT_NE(cpu_device, nullptr) << "No CPU device found";

  // Check compatibility - ExampleEP supports CPU, so should return no incompatibility reasons
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api->GetHardwareDeviceEpIncompatibilityDetails(env, Utils::example_ep_info.registration_name.c_str(),
                                                                     cpu_device, &details));
  ASSERT_NE(details, nullptr);

  // Verify compatible (no incompatibility reasons)
  uint32_t reasons_bitmask = 0xFFFFFFFF;
  ASSERT_ORTSTATUS_OK(api->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_EQ(reasons_bitmask, 0u) << "CPU device should be compatible with example_plugin_ep";

  int32_t error_code = -1;
  ASSERT_ORTSTATUS_OK(api->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, 0);

  api->ReleaseDeviceEpIncompatibilityDetails(details);
}

// Tests the GetHardwareDeviceEpIncompatibilityDetails C API with the example plugin EP.
// The example plugin EP only supports CPU devices, so this test verifies that a GPU device
// is reported as incompatible (reasons_bitmask != 0).
TEST(OrtEpLibrary, PluginEp_GpuDevice_ReturnsInCompatible) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = static_cast<OrtEnv*>(*ort_env);

  // Register the regular example plugin EP (CPU-only)
  RegisteredEpDeviceUniquePtr example_ep;
  ASSERT_NO_FATAL_FAILURE(Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, example_ep));

  // Get all hardware devices
  size_t num_hw_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetNumHardwareDevices(env, &num_hw_devices));
  ASSERT_GT(num_hw_devices, 0u);
  std::vector<const OrtHardwareDevice*> hw_devices(num_hw_devices);
  ASSERT_ORTSTATUS_OK(api->GetHardwareDevices(env, hw_devices.data(), num_hw_devices));

  // Find a GPU device using the public accessor
  const OrtHardwareDevice* gpu_device = nullptr;
  for (size_t i = 0; i < num_hw_devices; ++i) {
    if (api->HardwareDevice_Type(hw_devices[i]) == OrtHardwareDeviceType_GPU) {
      gpu_device = hw_devices[i];
      break;
    }
  }

  if (gpu_device == nullptr) {
    // GPU device not found, early exit
    GTEST_SKIP() << "No GPU device found";
  }

  // Check compatibility - ExampleEP only supports CPU, so GPU should return incompatibility reasons
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api->GetHardwareDeviceEpIncompatibilityDetails(env, Utils::example_ep_info.registration_name.c_str(),
                                                                     gpu_device, &details));
  ASSERT_NE(details, nullptr);

  // Verify incompatible (should have incompatibility reasons)
  uint32_t reasons_bitmask = 0;
  ASSERT_ORTSTATUS_OK(api->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_NE(reasons_bitmask, 0u) << "GPU device should be incompatible with example_plugin_ep (CPU-only)";

  api->ReleaseDeviceEpIncompatibilityDetails(details);
}
}  // namespace test
}  // namespace onnxruntime
