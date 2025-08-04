// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// registration/selection is only supported on windows as there's no device discovery on other platforms
#ifdef _WIN32

#include <filesystem>
// #include <absl/base/config.h>
#include <gsl/gsl>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"

#include "test/autoep/test_autoep_utils.h"
#include "test/shared_lib/utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

namespace {
void RunModelWithPluginEp(Ort::SessionOptions& session_options) {
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
}  // namespace

// Creates a session with the example plugin EP and runs a model with a single Mul node.
// Uses AppendExecutionProvider_V2 to append the example plugin EP to the session.
TEST(OrtEpLibrary, PluginEp_AppendV2_MulInference) {
  RegisteredEpDeviceUniquePtr example_ep;
  Utils::RegisterAndGetExampleEp(*ort_env, example_ep);
  const OrtEpDevice* plugin_ep_device = example_ep.get();

  // Create session with example plugin EP
  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(plugin_ep_device)}, ep_options);

  RunModelWithPluginEp(session_options);
}

// Creates a session with the example plugin EP and runs a model with a single Mul node.
// Uses the PREFER_CPU policy to append the example plugin EP to the session.
TEST(OrtEpLibrary, PluginEp_PreferCpu_MulInference) {
  RegisteredEpDeviceUniquePtr example_ep;
  Utils::RegisterAndGetExampleEp(*ort_env, example_ep);

  {
    // PREFER_CPU pick our example EP over ORT CPU EP. TODO: Actually assert this.
    Ort::SessionOptions session_options;
    session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_CPU);
    RunModelWithPluginEp(session_options);
  }
}

// Generate an EPContext model with a plugin EP.
// This test uses the OrtCompileApi but could also be done by setting the appropriate session option configs.
TEST(OrtEpLibrary, PluginEp_GenEpContextModel) {
  RegisteredEpDeviceUniquePtr example_ep;
  Utils::RegisterAndGetExampleEp(*ort_env, example_ep);
  const OrtEpDevice* plugin_ep_device = example_ep.get();

  {
    const ORTCHAR_T* input_model_file = ORT_TSTR("testdata/mul_1.onnx");
    const ORTCHAR_T* output_model_file = ORT_TSTR("plugin_ep_mul_1_ctx.onnx");
    std::filesystem::remove(output_model_file);

    // Create session with example plugin EP
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;

    session_options.AppendExecutionProvider_V2(*ort_env, {Ort::ConstEpDevice(plugin_ep_device)}, ep_options);

    // Create model compilation options from the session options.
    Ort::ModelCompilationOptions compile_options(*ort_env, session_options);
    compile_options.SetInputModelPath(input_model_file);
    compile_options.SetOutputModelPath(output_model_file);

    // Compile the model.
    Ort::Status status = Ort::CompileModel(*ort_env, compile_options);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

    // Make sure the compiled model was generated.
    ASSERT_TRUE(std::filesystem::exists(output_model_file));
  }
}
}  // namespace test
}  // namespace onnxruntime

#endif  // _WIN32
