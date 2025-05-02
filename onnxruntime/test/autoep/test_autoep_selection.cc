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
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test_allocator.h"
#include "test/shared_lib/utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {
void DefaultDeviceSelection(const std::string& ep_name, std::vector<const OrtEpDevice*>& devices) {
  const OrtApi* c_api = &Ort::GetApi();
  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices = 0;

  std::vector<OrtEpDevice*> selected_ep_device;
  ASSERT_ORTSTATUS_OK(c_api->GetEpDevices(*ort_env, &ep_devices, &num_devices));
  for (size_t i = 0; i < num_devices; ++i) {
    const OrtEpDevice* device = ep_devices[i];
    if (c_api->EpDevice_EpName(device) == ep_name) {
      devices.push_back(device);
      break;
    }
  }

  ASSERT_TRUE(!devices.empty()) << "No devices found with EP name of " << ep_name;
}

bool IsRegistered(const std::string& ep_name) {
  static std::unordered_set<std::string> registered_eps;
  if (registered_eps.count(ep_name) == 0) {
    registered_eps.insert(ep_name);
    return false;
  }

  return true;
}
}  // namespace

template <typename ModelOutputT, typename ModelInputT = float, typename InputT = Input<float>>
static void TestInference(Ort::Env& env, const std::basic_string<ORTCHAR_T>& model_uri,
                          const std::string& ep_to_select,
                          std::optional<std::filesystem::path> library_path,
                          const Ort::KeyValuePairs& ep_options,
                          const std::vector<InputT>& inputs,
                          const char* output_name,
                          const std::vector<int64_t>& expected_dims_y,
                          const std::vector<ModelOutputT>& expected_values_y,
                          bool auto_select = true,  // auto select vs SessionOptionsAppendExecutionProvider_V2
                          const std::function<void(std::vector<const OrtEpDevice*>&)>& select_devices = nullptr,
                          bool test_session_creation_only = false) {
  Ort::SessionOptions session_options;

  if (library_path && IsRegistered(ep_to_select) == false) {
    ASSERT_ORTSTATUS_OK(Ort::GetApi().RegisterExecutionProviderLibrary(env, ep_to_select.c_str(),
                                                                       library_path->c_str()));
  }

  if (auto_select) {
    // manually specify EP to select for now
    session_options.AddConfigEntry("test.ep_to_select", ep_to_select.c_str());

    // add the provider options to the session options with the required prefix
    const std::string option_prefix = OrtSessionOptions::GetProviderOptionPrefix(ep_to_select.c_str());
    std::vector<const char*> keys, values;
    ep_options.GetKeyValuePairs(keys, values);
    for (size_t i = 0, end = keys.size(); i < end; ++i) {
      // add the default value with prefix
      session_options.AddConfigEntry((option_prefix + keys[i]).c_str(), values[i]);
    }
  } else {
    std::vector<const OrtEpDevice*> devices;
    if (select_devices) {
      select_devices(devices);
    } else {
      // pick the first one assigned to the EP.
      DefaultDeviceSelection(ep_to_select, devices);
    }

    // C API. Test the C++ API because if it works the C API must also work.
    // ASSERT_ORTSTATUS_OK(Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
    //    session_options, env, devices.data(), devices.size(),
    //    provider_options.keys.data(), provider_options.values.data(), provider_options.entries.size()));
    std::vector<Ort::ConstEpDevice> ep_devices;
    ep_devices.reserve(devices.size());
    for (const auto* device : devices) {
      ep_devices.emplace_back(device);
    }

    session_options.AppendExecutionProvider_V2(*ort_env, ep_devices, ep_options);
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
                  const Ort::KeyValuePairs& provider_options = Ort::KeyValuePairs{},
                  const std::function<void(std::vector<const OrtEpDevice*>&)>& select_devices = nullptr) {
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
                         auto_select,
                         select_devices);
  };

  run_test(true);   // auto ep selection after session creation
  run_test(false);  // SessionOptionsAppendExecutionProvider_V2
}
}  // namespace

TEST(AutoEpSelection, CpuEP) {
  RunBasicTest(kCpuExecutionProvider, std::nullopt);
}

#if defined(USE_CUDA)
TEST(AutoEpSelection, CudaEP) {
  Ort::KeyValuePairs provider_options;
  provider_options.Add("prefer_nhwc", "1");
  RunBasicTest(kCudaExecutionProvider, "onnxruntime_providers_cuda", provider_options);
}
#endif

#if defined(USE_DML)
TEST(AutoEpSelection, DmlEP) {
  Ort::KeyValuePairs provider_options;
  provider_options.Add("disable_metacommands", "true");  // checking options are passed through

  const auto select_devices = [&](std::vector<const OrtEpDevice*>& devices) {
    const OrtApi* c_api = &Ort::GetApi();
    const OrtEpDevice* const* ep_devices = nullptr;
    size_t num_devices = 0;

    std::vector<OrtEpDevice*> selected_ep_device;
    ASSERT_ORTSTATUS_OK(c_api->GetEpDevices(*ort_env, &ep_devices, &num_devices));
    for (size_t i = 0; i < num_devices; ++i) {
      const OrtEpDevice* ep_device = ep_devices[i];
      if (strcmp(c_api->EpDevice_EpName(ep_device), kDmlExecutionProvider) == 0) {
        const auto* device = c_api->EpDevice_Device(ep_device);
        const OrtKeyValuePairs* kvps = c_api->HardwareDevice_Metadata(device);

        if (devices.empty()) {
          // add the first device
          devices.push_back(ep_device);
        } else {
          // if this is available, 0 == best performance
          auto* perf_index = c_api->GetKeyValue(kvps, "HighPerformanceIndex");
          if (perf_index && strcmp(perf_index, "0") == 0) {
            devices[0] = ep_device;  // replace as this is the higher performance device
          }
        }
      }
    }

    ASSERT_TRUE(!devices.empty()) << "No DML devices found";
  };

  RunBasicTest(kDmlExecutionProvider, std::nullopt, provider_options, select_devices);
}
#endif

#if defined(USE_WEBGPU)
TEST(AutoEpSelection, WebGpuEP) {
  RunBasicTest(kWebGpuExecutionProvider, std::nullopt);
}
#endif

// tests for AutoEP selection related things in the API that aren't covered by the other tests.
TEST(AutoEpSelection, MiscApiTests) {
  const OrtApi* c_api = &Ort::GetApi();

  // nullptr and empty input to OrtKeyValuePairs
  {
    OrtKeyValuePairs* kvps = nullptr;
    c_api->CreateKeyValuePairs(&kvps);
    c_api->AddKeyValuePair(kvps, "key1", nullptr);    // should be ignored
    c_api->AddKeyValuePair(kvps, nullptr, "value1");  // should be ignored
    c_api->RemoveKeyValuePair(kvps, nullptr);         // should be ignored

    c_api->AddKeyValuePair(kvps, "", "value2");  // empty key should be ignored
    ASSERT_EQ(c_api->GetKeyValue(kvps, ""), nullptr);

    c_api->AddKeyValuePair(kvps, "key2", "");  // empty value is allowed
    ASSERT_EQ(c_api->GetKeyValue(kvps, "key2"), std::string(""));

    c_api->ReleaseKeyValuePairs(kvps);
  }

  // construct KVP from std::unordered_map
  {
    std::unordered_map<std::string, std::string> kvps;
    kvps["key1"] = "value1";
    kvps["key2"] = "value2";
    Ort::KeyValuePairs ort_kvps(kvps);
    ASSERT_EQ(ort_kvps.GetValue("key1"), std::string("value1"));
    ASSERT_EQ(ort_kvps.GetValue("key2"), std::string("value2"));
  }

  std::vector<Ort::ConstEpDevice> ep_devices = ort_env->GetEpDevices();

  // explicit EP selection with Ort::KeyValuePairs for options
  {
    Ort::SessionOptions session_options;
    Ort::KeyValuePairs ep_options;
    ep_options.Add("option1", "true");
    session_options.AppendExecutionProvider_V2(*ort_env, {ep_devices[0]}, ep_options);
  }

  // explicit EP selection with <std::string, std::string> for options
  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    ep_options["option1"] = "true";
    session_options.AppendExecutionProvider_V2(*ort_env, {ep_devices[0]}, ep_options);
  }
}

namespace {
struct ExamplePluginInfo {
  const std::filesystem::path library_path =
#if _WIN32
      "example_plugin_ep.dll";
#else
      "libexample_plugin_ep.so";
#endif
  const std::string registration_name = "example_ep";
};

static const ExamplePluginInfo example_plugin_info;
}  // namespace

TEST(OrtEpLibrary, LoadUnloadPluginLibrary) {
  const std::filesystem::path& library_path = example_plugin_info.library_path;
  const std::string& registration_name = example_plugin_info.registration_name;

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

TEST(OrtEpLibrary, LoadUnloadPluginLibraryCxxApi) {
  const std::filesystem::path& library_path = example_plugin_info.library_path;
  const std::string& registration_name = example_plugin_info.registration_name;

  // this should load the library and create OrtEpDevice
  ort_env->RegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());

  std::vector<Ort::ConstEpDevice> ep_devices = ort_env->GetEpDevices();

  // should be one device for the example EP
  auto test_ep_device = std::find_if(ep_devices.begin(), ep_devices.end(),
                                     [&registration_name](Ort::ConstEpDevice& device) {
                                       // the example uses the registration name for the EP name
                                       // but that is not a requirement and the two can differ.
                                       return device.EpName() == registration_name;
                                     });
  ASSERT_NE(test_ep_device, ep_devices.end()) << "Expected an OrtEpDevice to have been created by the test library.";

  // test all the C++ getters. expected values are from \onnxruntime\test\autoep\library\example_plugin_ep.cc
  ASSERT_STREQ(test_ep_device->EpVendor(), "Contoso");

  auto metadata = test_ep_device->EpMetadata();
  ASSERT_STREQ(metadata.GetValue("version"), "0.1");

  auto options = test_ep_device->EpOptions();
  ASSERT_STREQ(options.GetValue("run_really_fast"), "true");

  // the CPU device info will vary by machine so check for the lowest common denominator values
  Ort::ConstHardwareDevice device = test_ep_device->Device();
  ASSERT_EQ(device.Type(), OrtHardwareDeviceType_CPU);
  ASSERT_GE(device.VendorId(), 0);
  ASSERT_GE(device.DeviceId(), 0);
  ASSERT_NE(device.Vendor(), nullptr);
  Ort::ConstKeyValuePairs device_metadata = device.Metadata();
  std::unordered_map<std::string, std::string> metadata_entries = device_metadata.GetKeyValuePairs();
  ASSERT_GT(metadata_entries.size(), 0);  // should have at least SPDRP_HARDWAREID on Windows

  // and this should unload it without throwing
  ort_env->UnregisterExecutionProviderLibrary(registration_name.c_str());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // _WIN32
