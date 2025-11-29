// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
// #include <absl/base/config.h>
#include <gsl/gsl>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

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

void RunPartiallySupportedModelWithPluginEp(const Ort::SessionOptions& session_options) {
  // This model has Add -> Mul -> Add. The example plugin EP only supports Mul.
  Ort::Session session(*ort_env, ORT_TSTR("testdata/add_mul_add.onnx"), session_options);

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
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  RunPartiallySupportedModelWithPluginEp(session_options);
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

  // Create session with example kernel-based plugin EP
  Ort::SessionOptions session_options;
  session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");  // Fail if any node assigned to CPU EP.

  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {plugin_ep_device}, ep_options);

  // This model has Squeeze, Mul, and Relu nodes. The example plugin EP supports all nodes using registered kernels.
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
}  // namespace test
}  // namespace onnxruntime
