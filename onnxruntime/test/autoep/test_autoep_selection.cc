// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// registration/selection is only supported on windows as there's no device discovery on other platforms
#ifdef _WIN32

#include <filesystem>
#include <absl/base/config.h>
#include <gtest/gtest.h>

#include "core/common/common.h"
#include "core/framework/provider_options.h"
#include "core/graph/constants.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test_allocator.h"
#include "test/shared_lib/utils.h"
#include "test/util/include/asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

template <typename ModelOutputT, typename ModelInputT = float, typename InputT = Input<float>>
static void TestInference(Ort::Env& env, const std::basic_string<ORTCHAR_T>& model_uri,
                          const std::string& ep_to_select,
                          std::optional<std::filesystem::path> library_path,
                          const OrtKeyValuePairs& provider_options,
                          const std::vector<InputT>& inputs,
                          const char* output_name,
                          const std::vector<int64_t>& expected_dims_y,
                          const std::vector<ModelOutputT>& expected_values_y,
                          bool auto_select = true,  // auto select vs SessionOptionsAppendExecutionProvider_V2
                          bool test_session_creation_only = false) {
  Ort::SessionOptions session_options;

  if (library_path) {
    // use EP name as registration name for now. there's some hardcoded matching of names to special case
    // the provider bridge EPs short term.
    ASSERT_ORTSTATUS_OK(Ort::GetApi().RegisterExecutionProviderLibrary(env, ep_to_select.c_str(),
                                                                       library_path->c_str()));
  }

  if (auto_select) {
    // manually specify EP to select for now
    ASSERT_ORTSTATUS_OK(Ort::GetApi().AddSessionConfigEntry(session_options, "test.ep_to_select",
                                                            ep_to_select.c_str()));

    const std::string option_prefix = ProviderOptionsUtils::GetProviderOptionPrefix(ep_to_select);
    for (const auto& [key, value] : provider_options.entries) {
      // add the default value with prefix
      session_options.AddConfigEntry((option_prefix + key).c_str(), value.c_str());
    }
  } else {
    ASSERT_ORTSTATUS_OK(Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
        session_options, env, ep_to_select.c_str(),
        provider_options.keys.data(), provider_options.values.data(), provider_options.entries.size()));
  }

  // if session creation passes, model loads fine
  Ort::Session session(env, model_uri.c_str(), session_options);

  // caller wants to test running the model (not just loading the model)
  if (!test_session_creation_only) {
    auto default_allocator = std::make_unique<MockedOrtAllocator>();
    RunSession<ModelOutputT, ModelInputT, InputT>(default_allocator.get(),
                                                  session,
                                                  inputs,
                                                  output_name,
                                                  expected_dims_y,
                                                  expected_values_y,
                                                  nullptr);
  }
}

namespace {
void RunBasicTest(const std::string& ep_name, std::optional<std::filesystem::path> library_path,
                  const OrtKeyValuePairs& provider_options = {}) {
  const auto run_test = [&](bool auto_select) {
    std::vector<Input<float>> inputs(1);
    auto& input = inputs.back();
    input.name = "X";
    input.dims = {3, 2};
    input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    // prepare expected inputs and outputs
    std::vector<int64_t> expected_dims_y = {3, 2};
    std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
    TestInference<float>(*ort_env, ORT_TSTR("testdata/mul_1.onnx"),
                         ep_name,
                         library_path,
                         provider_options,
                         inputs,
                         "Y",
                         expected_dims_y,
                         expected_values_y,
                         auto_select);
  };

  run_test(true);   // auto ep selection after session creation
  run_test(false);  // SessionOptionsAppendExecutionProvider_V2
}
}  // namespace

TEST(AutoEpSelection, CpuEP) {
  RunBasicTest(std::string(kCpuExecutionProvider), std::nullopt);
}

#if defined(USE_CUDA)
TEST(AutoEpSelection, CudaEP) {
  OrtKeyValuePairs provider_options;
  provider_options.Add("prefer_nhwc", "1");
  RunBasicTest("CUDA", "onnxruntime_providers_cuda", provider_options);
}
#endif

#if defined(USE_DML)
TEST(AutoEpSelection, DmlEP) {
  OrtKeyValuePairs provider_options;
  provider_options.Add("device_id", "0");
  RunBasicTest("DML", std::nullopt, provider_options);
}
#endif

#if defined(USE_WEBGPU)
TEST(AutoEpSelection, WebGpuEP) {
  RunBasicTest("WebGPU", std::nullopt);
}
#endif

TEST(OrtEpLibrary, LoadUnloadPluginLibrary) {
  std::filesystem::path library_path = "example_plugin_ep_library.dll";
  const std::string registration_name = "example_ep";

  Ort::SessionOptions session_options;

  OrtEnv* c_api_env = *ort_env;
  const OrtApi* c_api = &Ort::GetApi();
  // this should load the library and create OrtEpDevice
  ASSERT_ORTSTATUS_OK(Ort::GetApi().RegisterExecutionProviderLibrary(c_api_env, registration_name.c_str(),
                                                                     library_path.c_str()));

  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices = 0;

  ASSERT_ORTSTATUS_OK(Ort::GetApi().GetEpDevices(c_api_env, &ep_devices, &num_devices));
  // should be one device for the example EP
  auto num_test_ep_devices = std::count_if(ep_devices, ep_devices + num_devices,
                                           [&registration_name, &c_api](const OrtEpDevice* device) {
                                             // the example uses the registration name for the EP name
                                             // but that is not a requirement and the two can differ.
                                             return c_api->EpDevice_EpName(device) == registration_name;
                                           });
  ASSERT_EQ(num_test_ep_devices, 1) << "Expected an OrtEpDevice to have been created by the test library.";

  // and this should unload it
  ASSERT_ORTSTATUS_OK(Ort::GetApi().UnregisterExecutionProviderLibrary(c_api_env,
                                                                       registration_name.c_str()));
}
}  // namespace test
}  // namespace onnxruntime

#endif  // _WIN32
