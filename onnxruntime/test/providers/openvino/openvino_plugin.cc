#include <filesystem>
#include <gsl/span>

#include "gtest/gtest.h"
#include "core/common/common.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "onnxruntime_cxx_api.h"
#include "api_asserts.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

extern std::unique_ptr<Ort::Env> ort_env;

struct OrtEpLibraryOv : public ::testing::Test {
  static const inline std::filesystem::path library_path =
#if _WIN32
      "onnxruntime_providers_openvino.dll";
#else
      "libonnxruntime_providers_openvino.so";
#endif
  static const inline std::string registration_name = "OpenVINOExecutionProvider";

  void SetUp() override {
#ifndef _WIN32
    GTEST_SKIP() << "Skipping OpenVINO EP tests as the OpenVINO plugin is not built.";
#endif
    ort_env->RegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());
  }

  void TearDown() override {
#ifndef _WIN32
    GTEST_SKIP() << "Skipping OpenVINO EP tests as the OpenVINO plugin is not built.";
#endif
    ort_env->UnregisterExecutionProviderLibrary(registration_name.c_str());
  }

  void RunModelWithSession(Ort::Session& session) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<int64_t> shape = {3, 2};
    std::vector<float> input0_data(6, 2.0f);
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> ort_input_names;
    ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, input0_data.data(), input0_data.size(), shape.data(), shape.size()));
    ort_input_names.push_back("X");
    std::array<const char*, 1> output_names{"Y"};
    std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                      ort_inputs.size(), output_names.data(), output_names.size());
    Ort::Value& ort_output = ort_outputs[0];
    const float* output_data = ort_output.GetTensorData<float>();
    gsl::span<const float> output_span(output_data, 6);
    EXPECT_THAT(output_span, ::testing::ElementsAre(2, 4, 6, 8, 10, 12));
  }

  void RunModelWithPluginEp(Ort::SessionOptions& session_options) {
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
    RunModelWithSession(session);
  }

  void GenerateEpContextOnLegacyPath(std::filesystem::path epctx, bool embed_mode) {
    Ort::SessionOptions session_options{};
    std::filesystem::remove(epctx);
    // Add config option to enable EP context
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, epctx.string().c_str());
    session_options.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, embed_mode ? "1" : "0");
    session_options.AppendExecutionProvider_OpenVINO_V2({{"device_type", "CPU"}});
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
    RunModelWithSession(session);
  }

  void GenerateEpContextOnPluginPath(std::filesystem::path epctx, bool embed_mode) {
    Ort::SessionOptions session_options{};
    std::filesystem::remove(epctx);
    // Add config option to enable EP context
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
    session_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    session_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath, epctx.string().c_str());
    session_options.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, embed_mode ? "1" : "0");
    Ort::ConstEpDevice plugin_ep_device = GetOvCpuEpDevice();
    ASSERT_NE(plugin_ep_device, nullptr);
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, std::vector<Ort::ConstEpDevice>{plugin_ep_device}, ep_options);
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
    RunModelWithSession(session);
  }

  Ort::ConstEpDevice GetOvCpuEpDevice(std::string device_type = "CPU") {
    auto ep_devices = ort_env->GetEpDevices();
    Ort::ConstEpDevice plugin_ep_device{};

    for (Ort::ConstEpDevice& device : ep_devices) {
      if (device.Device().Type() == OrtHardwareDeviceType_CPU &&
          std::string_view(device.EpName()).find(registration_name) != std::string::npos) {
        const auto& meta_kv = device.EpMetadata().GetKeyValuePairs();
        auto device_type_it = meta_kv.find("ov_device");
        if (device_type_it != meta_kv.end()) {
          if (device_type_it->second == device_type) {
            plugin_ep_device = device;
            break;
          }
        }
      }
    }

    return plugin_ep_device;
  }
};

TEST_F(OrtEpLibraryOv, LoadUnloadPluginLibrary) {
  auto ep_devices = ort_env->GetEpDevices();
  auto test_cpu_ep_device = GetOvCpuEpDevice();
  ASSERT_NE(test_cpu_ep_device, nullptr);
  ASSERT_STREQ(test_cpu_ep_device.EpVendor(), "Intel");
  Ort::ConstHardwareDevice device = test_cpu_ep_device.Device();
  ASSERT_EQ(device.Type(), OrtHardwareDeviceType_CPU);
  ASSERT_GE(device.VendorId(), 0);
  ASSERT_GE(device.DeviceId(), 0);
  ASSERT_NE(device.Vendor(), nullptr);
  std::unordered_map<std::string, std::string> ep_metadata_entries = test_cpu_ep_device.EpMetadata().GetKeyValuePairs();
  ASSERT_GT(ep_metadata_entries.size(), 0);
  ASSERT_GT(ep_metadata_entries.count("ov_device"), 0);
}

TEST_F(OrtEpLibraryOv, MetaDevicesAvailable) {
  auto ep_devices = ort_env->GetEpDevices();
  auto expected_meta_devices = {"AUTO"};

  for (auto& expected_meta_device : expected_meta_devices) {
    std::string expected_ep_name = registration_name + "." + expected_meta_device;
    auto it = std::find_if(ep_devices.begin(), ep_devices.end(),
                           [&](Ort::ConstEpDevice& device) {
                             return std::string_view(device.EpName()).find(expected_ep_name) != std::string::npos;
                           });
    bool meta_device_found = it != ep_devices.end();
    ASSERT_TRUE(meta_device_found) << "Expected to find " << expected_ep_name;
  }
}

TEST_F(OrtEpLibraryOv, RunSessionWithAllAUTODevices) {
  auto ep_devices = ort_env->GetEpDevices();
  std::vector<Ort::ConstEpDevice> matching_devices;

  for (const auto& device : ep_devices) {
    std::string ep_name = device.EpName();
    if (ep_name.find(registration_name) != std::string::npos &&
        (ep_name == registration_name + ".AUTO")) {
      matching_devices.push_back(device);
    }
  }
  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_V2(*ort_env, matching_devices, std::unordered_map<std::string, std::string>{});
  Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
}

TEST_F(OrtEpLibraryOv, PluginEp_AppendV2_MulInference) {
  auto plugin_ep_device = GetOvCpuEpDevice();
  ASSERT_NE(plugin_ep_device, nullptr);

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> ep_options;
  session_options.AppendExecutionProvider_V2(*ort_env, std::vector<Ort::ConstEpDevice>{plugin_ep_device}, ep_options);
  RunModelWithPluginEp(session_options);
}

TEST_F(OrtEpLibraryOv, PluginEp_PreferCpu_MulInference) {
  Ort::SessionOptions session_options;
  session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_CPU);
  RunModelWithPluginEp(session_options);
}

struct EpCtxTestCases {
  const ORTCHAR_T* ctx_filename;
  bool embed_mode;
};

static const std::vector<EpCtxTestCases> ep_context_cases = {
    {ORT_TSTR("mul_1_ctx_cpu_embed1.onnx"), true},
    {ORT_TSTR("mul_1_ctx_cpu_embed0.onnx"), false},
    {ORT_TSTR("testdata/mul_1_ctx_cpu_embed0.onnx"), false}};

TEST_F(OrtEpLibraryOv, PluginEp_AppendV2_cpu_epctx_variants) {
  auto plugin_ep_device = GetOvCpuEpDevice();
  ASSERT_NE(plugin_ep_device, nullptr);

  for (const auto& test_case : ep_context_cases) {
    GenerateEpContextOnLegacyPath(test_case.ctx_filename, test_case.embed_mode);

    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, std::vector<Ort::ConstEpDevice>{plugin_ep_device}, ep_options);
    Ort::Session session(*ort_env, test_case.ctx_filename, session_options);
    RunModelWithSession(session);
  }
}

TEST_F(OrtEpLibraryOv, PluginEp_CheckV2DisallowedProviderOptions) {
  auto plugin_ep_device = GetOvCpuEpDevice();
  ASSERT_NE(plugin_ep_device, nullptr);
  std::vector<std::unordered_map<std::string, std::string>> disallowed_provider_option_examples = {
      {{"device_type", "CPU"}},
      {{"device_id", "CPU"}},
      {{"device_luid", "1234"}},
      {{"cache_dir", "cache"}},
      {{"precision", "F32"}},
      {{"context", "4"}},
      {{"num_of_threads", "1"}},
      {{"model_priority", "DEFAULT"}},
      {{"num_streams", "1"}},
      {{"enable_opencl_throttling", "true"}},
      {{"enable_qdq_optimizer", "true"}},
      {{"disable_dynamic_shapes", "true"}},
  };
  for (auto& example : disallowed_provider_option_examples) {
    EXPECT_THROW({
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_V2(*ort_env, std::vector<Ort::ConstEpDevice>{plugin_ep_device}, example);
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options); }, Ort::Exception);
  }
}

TEST_F(OrtEpLibraryOv, GenerateEpContextEmbedded) {
  GenerateEpContextOnPluginPath(ORT_TSTR("mul_1_ctx_cpu_embed1.onnx"), true);
}

TEST_F(OrtEpLibraryOv, GenerateEpContext) {
  GenerateEpContextOnPluginPath(ORT_TSTR("mul_1_ctx_cpu_embed0.onnx"), false);
}

TEST_F(OrtEpLibraryOv, PluginEp_AppendV2_cpu_epctx_plugin_roundtrip_variants) {
  auto plugin_ep_device = GetOvCpuEpDevice();
  ASSERT_NE(plugin_ep_device, nullptr);

  for (const auto& test_case : ep_context_cases) {
    if (test_case.embed_mode) {
      // TODO(ericcraw) Re-enable.
      // Skip the embed mode until upstream fix.
      continue;
    }

    GenerateEpContextOnPluginPath(test_case.ctx_filename, test_case.embed_mode);

    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, std::vector<Ort::ConstEpDevice>{plugin_ep_device}, ep_options);
    Ort::Session session(*ort_env, test_case.ctx_filename, session_options);
    RunModelWithSession(session);
  }
}

TEST_F(OrtEpLibraryOv, PluginEp_AppendV2_cpu_epctx_plugin_roundtrip_variants_absolute) {
  auto plugin_ep_device = GetOvCpuEpDevice();
  ASSERT_NE(plugin_ep_device, nullptr);

  for (const auto& test_case : ep_context_cases) {
    if (test_case.embed_mode) {
      // TODO(ericcraw) Re-enable.
      // Skip the embed mode until upstream fix.
      continue;
    }

    auto absolute_path = std::filesystem::absolute(test_case.ctx_filename).native();
    GenerateEpContextOnPluginPath(absolute_path.c_str(), test_case.embed_mode);

    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> ep_options;
    session_options.AppendExecutionProvider_V2(*ort_env, std::vector<Ort::ConstEpDevice>{plugin_ep_device}, ep_options);
    Ort::Session session(*ort_env, absolute_path.c_str(), session_options);
    RunModelWithSession(session);
  }
}

TEST_F(OrtEpLibraryOv, PluginEp_AppendV2_multiple_devices) {
  auto plugin_ep_device = GetOvCpuEpDevice();
  ASSERT_NE(plugin_ep_device, nullptr);

  std::vector<Ort::ConstEpDevice> multi_device_list(2, plugin_ep_device);  // 2 copies of cpu device.

  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_V2(*ort_env, multi_device_list, std::unordered_map<std::string, std::string>{});
  Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
}

TEST_F(OrtEpLibraryOv, PluginEp_AppendV2_mixed_factory_devices_throw_exception) {
  auto ep_devices = ort_env->GetEpDevices();
  std::vector<Ort::ConstEpDevice> matching_devices;

  for (const auto& device : ep_devices) {
    std::string ep_name = device.EpName();
    if (ep_name.find(registration_name) != std::string::npos &&
        (ep_name == registration_name || ep_name == registration_name + ".AUTO")) {
      matching_devices.push_back(device);
    }
  }

  ASSERT_GT(matching_devices.size(), 1) << "Expected more than one matching EP device";

  EXPECT_THROW({
        Ort::SessionOptions session_options;
        session_options.AppendExecutionProvider_V2(*ort_env, matching_devices, std::unordered_map<std::string, std::string>{});
        Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options); }, Ort::Exception);
}
