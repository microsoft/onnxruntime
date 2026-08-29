// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <filesystem>
#include <initializer_list>
#include <string_view>
// #include <absl/base/config.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "core/framework/provider_options.h"
#include "core/graph/constants.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#if defined(USE_COREML) && defined(__APPLE__)
#include <gsl/gsl>

#include "core/common/inlined_containers.h"
#include "core/providers/coreml/coreml_provider_factory.h"
#include "core/providers/coreml/model/host_utils.h"
#include "core/session/plugin_ep/ep_factory_coreml.h"
#endif

#include "test_allocator.h"
#include "test/autoep/test_autoep_utils.h"
#include "test/shared_lib/utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"
#include "test/util/include/file_util.h"

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
      const auto* hw_device = c_api->EpDevice_Device(device);
      const OrtKeyValuePairs* hw_kvps = c_api->HardwareDevice_Metadata(hw_device);

      const char* is_virtual = c_api->GetKeyValue(hw_kvps, kOrtHardwareDevice_MetadataKey_IsVirtual);
      ASSERT_TRUE(is_virtual == nullptr || strcmp(is_virtual, "0") == 0);

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
                          // manual select using functor
                          const std::function<void(std::vector<const OrtEpDevice*>&)>& select_devices = nullptr,
                          // auto select using policy
                          std::optional<OrtExecutionProviderDevicePolicy> policy = std::nullopt,
                          std::optional<EpSelectionDelegate> delegate = std::nullopt,
                          bool test_session_creation_only = false,
                          // If true, disables fallback of unsupported graph nodes to the ORT CPU
                          // EP. Session creation fails unless the selected non-CPU EP supports the
                          // entire graph. Setting this to true while explicitly selecting the ORT
                          // CPU EP is invalid and causes session creation to fail.
                          bool disable_cpu_ep_fallback = false,
                          // Optional callback invoked after session creation and before inference,
                          // for example to verify the session's EP assignment.
                          const std::function<void(Ort::Session&)>& session_checker = nullptr) {
  Ort::SessionOptions session_options;

  if (disable_cpu_ep_fallback) {
    session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  }

  if (library_path && IsRegistered(ep_to_select) == false) {
    ASSERT_ORTSTATUS_OK(Ort::GetApi().RegisterExecutionProviderLibrary(env, ep_to_select.c_str(),
                                                                       library_path->c_str()));
  }

  if (auto_select) {
    if (delegate) {
      session_options.SetEpSelectionPolicy(*delegate, nullptr);
    } else if (policy) {
      session_options.SetEpSelectionPolicy(*policy);
    } else {
      // manually specify EP to select
      session_options.AddConfigEntry("test.ep_to_select", ep_to_select.c_str());

      // add the provider options to the session options with the required prefix
      const std::string option_prefix = OrtSessionOptions::GetProviderOptionPrefix(ep_to_select.c_str());
      std::vector<const char*> keys, values;
      ep_options.GetKeyValuePairs(keys, values);
      for (size_t i = 0, end = keys.size(); i < end; ++i) {
        // add the default value with prefix
        session_options.AddConfigEntry((option_prefix + keys[i]).c_str(), values[i]);
      }
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
    //    provider_options.Keys().data(), provider_options.Values().data(), provider_options.Entries().size()));
    std::vector<Ort::ConstEpDevice> ep_devices;
    ep_devices.reserve(devices.size());
    for (const auto* device : devices) {
      ep_devices.emplace_back(device);
    }

    session_options.AppendExecutionProvider_V2(*ort_env, ep_devices, ep_options);
  }

  // if session creation passes, model loads fine
  Ort::Session session(env, model_uri.c_str(), session_options);

  if (session_checker) {
    // Stop this helper if session_checker reports a fatal assertion, rather than continuing to RunSession.
    ASSERT_NO_FATAL_FAILURE(session_checker(session));
  }

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
                  const std::function<void(std::vector<const OrtEpDevice*>&)>& select_devices = nullptr,
                  bool test_auto_select = true,
                  // Optional callback for the session created through AppendExecutionProvider_V2.
                  const std::function<void(Ort::Session&)>& v2_session_checker = nullptr,
                  // Applied to both paths. In the AppendExecutionProvider_V2 path, v2_session_checker verifies the
                  // input assignment directly, while disabling CPU fallback requires the selected EP to handle the
                  // entire graph. The "test.ep_to_select" path has no session checker, so disabling CPU fallback
                  // prevents a false pass if any part of the graph is assigned to the ORT CPU EP instead of the
                  // selected EP.
                  bool disable_cpu_ep_fallback = false) {
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
                         select_devices,
                         /*policy*/ std::nullopt,
                         /*delegate*/ std::nullopt,
                         /*test_session_creation_only*/ false,
                         disable_cpu_ep_fallback,
                         auto_select ? nullptr : v2_session_checker);
  };

  if (test_auto_select) {
    run_test(true);  // auto ep selection after session creation
  }

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
  const auto cuda_ep_lib_path =
      std::filesystem::path{GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_cuda"))};
  RunBasicTest(kCudaExecutionProvider, cuda_ep_lib_path, provider_options);
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

        const char* is_virtual = c_api->GetKeyValue(kvps, kOrtHardwareDevice_MetadataKey_IsVirtual);
        ASSERT_TRUE(is_virtual == nullptr || strcmp(is_virtual, "0") == 0);

        if (devices.empty()) {
          // add the first device
          devices.push_back(ep_device);
        } else {
          // if this is available, 0 == best performance
          auto* perf_index = c_api->GetKeyValue(kvps, "DxgiHighPerformanceIndex");
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

#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
TEST(AutoEpSelection, WebGpuEP) {
  RunBasicTest(kWebGpuExecutionProvider, std::nullopt);
}
#endif  // defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)

#if defined(USE_COREML) && defined(__APPLE__)
namespace {
// The CoreML EP factory creates OrtEpDevice instances only for claimable NPU and GPU hardware. The tests use
// GetHardwareDevices and the runtime Core ML version to determine which devices the factory should advertise,
// rather than relying on GetEpDevices itself. A device-dependent test skips only if the required hardware is
// unavailable or excluded by the Core ML version check. Otherwise, the test fails if the factory is not registered
// or does not return the expected OrtEpDevice. The tests also verify that no devices are advertised below Core ML 5
// and that no NPU is advertised below Core ML 6.
struct CoreMLDevices {
  const OrtEpDevice* npu = nullptr;
  const OrtEpDevice* gpu = nullptr;
};

// Hardware device types reported by device discovery, independent of EP support.
struct HardwareDeviceTypes {
  bool has_npu = false;
  bool has_gpu = false;
};

HardwareDeviceTypes GetHardwareDeviceTypes() {
  HardwareDeviceTypes found;
  const OrtApi* c_api = &Ort::GetApi();
  size_t num_devices = 0;

  Ort::ThrowOnError(c_api->GetNumHardwareDevices(*ort_env, &num_devices));
  if (num_devices == 0) {
    return found;
  }

  InlinedVector<const OrtHardwareDevice*> devices(num_devices);
  Ort::ThrowOnError(c_api->GetHardwareDevices(*ort_env, devices.data(), num_devices));

  for (const OrtHardwareDevice* device : devices) {
    const OrtHardwareDeviceType type = c_api->HardwareDevice_Type(device);
    found.has_npu |= type == OrtHardwareDeviceType_NPU;
    found.has_gpu |= type == OrtHardwareDeviceType_GPU;
  }

  return found;
}

CoreMLDevices GetCoreMLEpDevices() {
  CoreMLDevices found;
  size_t num_npu = 0;
  size_t num_gpu = 0;
  const OrtApi* c_api = &Ort::GetApi();
  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices = 0;

  Ort::ThrowOnError(c_api->GetEpDevices(*ort_env, &ep_devices, &num_devices));

  for (size_t i = 0; i < num_devices; ++i) {
    const OrtEpDevice* ep_device = ep_devices[i];
    if (strcmp(c_api->EpDevice_EpName(ep_device), kCoreMLExecutionProvider) != 0) {
      continue;
    }

    const OrtHardwareDeviceType type = c_api->HardwareDevice_Type(c_api->EpDevice_Device(ep_device));
    if (type == OrtHardwareDeviceType_NPU) {
      ++num_npu;
      if (found.npu == nullptr) {
        found.npu = ep_device;
      }
    } else if (type == OrtHardwareDeviceType_GPU) {
      ++num_gpu;
      if (found.gpu == nullptr) {
        found.gpu = ep_device;
      }
    } else {
      ADD_FAILURE() << "CoreML EP advertised an OrtEpDevice with unexpected hardware device type " << type;
    }
  }

  // The factory may advertise at most one NPU and one GPU. This helper retains only the first OrtEpDevice of each
  // type, so the separate counters ensure that any additional OrtEpDevice of the same type is detected.
  EXPECT_LE(num_npu, 1u) << "CoreML EP advertised more than one NPU OrtEpDevice.";
  EXPECT_LE(num_gpu, 1u) << "CoreML EP advertised more than one GPU OrtEpDevice.";

  return found;
}

// Finds the first CoreML OrtEpDevice returned by GetEpDevices, or nullptr if none is advertised.
const OrtEpDevice* GetFirstCoreMLEpDevice() {
  const OrtApi* c_api = &Ort::GetApi();
  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices = 0;

  Ort::ThrowOnError(c_api->GetEpDevices(*ort_env, &ep_devices, &num_devices));

  for (size_t i = 0; i < num_devices; ++i) {
    if (strcmp(c_api->EpDevice_EpName(ep_devices[i]), kCoreMLExecutionProvider) == 0) {
      return ep_devices[i];
    }
  }

  return nullptr;
}

// Returns whether this machine has an NPU that the CoreML EP factory may advertise at the runtime Core ML version.
bool CoreMLCanClaimNpu(const HardwareDeviceTypes& hardware) {
  return hardware.has_npu &&
         CoreMLEpFactory::CanClaimDeviceType(OrtHardwareDeviceType_NPU, coreml::util::CoreMLVersion());
}

// Returns whether this machine has a GPU that the CoreML EP factory may advertise at the runtime Core ML version.
bool CoreMLCanClaimGpu(const HardwareDeviceTypes& hardware) {
  return hardware.has_gpu &&
         CoreMLEpFactory::CanClaimDeviceType(OrtHardwareDeviceType_GPU, coreml::util::CoreMLVersion());
}

// GetErrorCode can be called only on a failed Ort::Status. An OK status contains a null OrtStatus*. Verify failure
// first so the helper stops before GetErrorCode if the call unexpectedly succeeds.
// SCOPED_TRACE includes 'selection' in failure messages, making it clear which input failed.
void ExpectInvalidArgument(const Ort::Status& status, std::string_view selection) {
  SCOPED_TRACE(selection);
  ASSERT_FALSE(status.IsOK()) << "Expected ORT_INVALID_ARGUMENT, but the call succeeded.";
  EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
}

// Verifies that a session for mul_1.onnx assigns its input to the CoreML EP and produces the expected output.
void AssertMul1SessionRunsOnCoreML(Ort::Session& session) {
  const OrtApi* c_api = &Ort::GetApi();
  const OrtEpDevice* input_ep_device = nullptr;
  ASSERT_ORTSTATUS_OK(c_api->SessionGetEpDeviceForInputs(session, &input_ep_device, 1));
  ASSERT_NE(input_ep_device, nullptr);
  ASSERT_STREQ(c_api->EpDevice_EpName(input_ep_device), kCoreMLExecutionProvider);

  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  auto allocator = std::make_unique<MockedOrtAllocator>();
  RunSession<float>(allocator.get(), session, inputs, "Y",
                    {3, 2}, {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f}, nullptr);
}
}  // namespace

TEST(AutoEpSelection, CoreMLEP) {
  const HardwareDeviceTypes hardware = GetHardwareDeviceTypes();
  const bool can_claim_npu = CoreMLCanClaimNpu(hardware);
  const bool can_claim_gpu = CoreMLCanClaimGpu(hardware);
  const CoreMLDevices coreml_devices = GetCoreMLEpDevices();

  // Verify that the factory does not advertise devices rejected by its version requirements.
  if (hardware.has_npu && !can_claim_npu) {
    EXPECT_EQ(coreml_devices.npu, nullptr) << "CoreML EP must not advertise the NPU below Core ML 6.";
  }
  if (hardware.has_gpu && !can_claim_gpu) {
    EXPECT_EQ(coreml_devices.gpu, nullptr) << "CoreML EP must not advertise devices below Core ML 5.";
  }

  // Skip when the factory cannot advertise any discovered accelerator. For example, on iOS versions earlier
  // than 16, discovery may report an NPU that cannot be advertised because NPU selection requires Core ML 6,
  // while GPU discovery does not currently support iOS.
  if (!can_claim_npu && !can_claim_gpu) {
    GTEST_SKIP() << "No hardware device that the CoreML EP factory can advertise on this machine.";
  }

  if (can_claim_npu) {
    ASSERT_NE(coreml_devices.npu, nullptr) << "CoreML EP did not claim the NPU hardware device.";
  }
  if (can_claim_gpu) {
    ASSERT_NE(coreml_devices.gpu, nullptr) << "CoreML EP did not claim the GPU hardware device.";
  }

  // Verify the CoreML factory's vendor name and vendor ID, along with the hardware vendor ID. The factory reports
  // "Microsoft", like the other internal factories: the factory vendor identifies who provides the EP
  // implementation, not the hardware vendor.
  // Apple device discovery currently assigns Apple's PCI vendor id (0x106B) to every NPU and GPU it reports. The
  // CoreML factory intentionally accepts any discovered GPU regardless of vendor, so this hardware vendor ID
  // assertion must be updated if discovery later reports a non-Apple GPU (see the multi-GPU TODO in
  // core/platform/apple/device_discovery.cc).
  const OrtApi* c_api = &Ort::GetApi();
  size_t num_ep_devices = 0;
  const OrtEpDevice* const* all_ep_devices = nullptr;
  ASSERT_ORTSTATUS_OK(c_api->GetEpDevices(*ort_env, &all_ep_devices, &num_ep_devices));

  for (size_t i = 0; i < num_ep_devices; ++i) {
    const OrtEpDevice* ep_device = all_ep_devices[i];
    if (strcmp(c_api->EpDevice_EpName(ep_device), kCoreMLExecutionProvider) != 0) {
      continue;
    }

    EXPECT_STREQ(c_api->EpDevice_EpVendor(ep_device), "Microsoft");
    EXPECT_EQ(c_api->HardwareDevice_VendorId(c_api->EpDevice_Device(ep_device)), uint32_t{0x106B});
    EXPECT_EQ(ep_device->ep_factory->GetVendorId(ep_device->ep_factory), OrtDevice::VendorIds::MICROSOFT);
  }

  // Both the "test.ep_to_select" auto-selection path and the AppendExecutionProvider_V2 path select CoreML as
  // the only non-CPU EP. Disabling CPU fallback makes session creation fail unless CoreML takes the entire graph.
  // The V2 path also checks the input assignment directly. CoreMLEPPreferNpu and CoreMLEPPreferGpu perform the
  // same assignment check for policy-based selection.
  RunBasicTest(kCoreMLExecutionProvider, std::nullopt, Ort::KeyValuePairs{}, /*select_devices*/ nullptr,
               /*test_auto_select*/ true, AssertMul1SessionRunsOnCoreML, /*disable_cpu_ep_fallback*/ true);
}

// Select the CoreML NPU and GPU in a single AppendExecutionProvider_V2 call without specifying MLComputeUnits.
// The factory must derive MLComputeUnits=ALL from the selected devices, and session creation must succeed.
TEST(AutoEpSelection, CoreMLEPMultipleDevices) {
  const HardwareDeviceTypes hardware = GetHardwareDeviceTypes();
  if (!CoreMLCanClaimNpu(hardware) || !CoreMLCanClaimGpu(hardware)) {
    GTEST_SKIP() << "Need both an NPU and a GPU that the CoreML EP can advertise on this machine.";
  }

  const CoreMLDevices coreml_devices = GetCoreMLEpDevices();
  ASSERT_NE(coreml_devices.npu, nullptr) << "CoreML EP did not claim the NPU hardware device.";
  ASSERT_NE(coreml_devices.gpu, nullptr) << "CoreML EP did not claim the GPU hardware device.";

  const auto select_devices = [&](std::vector<const OrtEpDevice*>& devices) {
    devices = {coreml_devices.npu, coreml_devices.gpu};
  };

  // MLComputeUnits is intentionally omitted. If ALL were supplied explicitly, the factory would validate it instead
  // of deriving it from the selected NPU and GPU. The separate CoreMLEPDefaultComputeUnits test checks this
  // derivation directly. This test verifies the same behavior during session creation.
  // CoreMLEPNarrowingComputeUnitsOverride covers the same multi-device V2 path with non-empty ep_options.
  // select_devices applies only to the AppendExecutionProvider_V2 path. The "test.ep_to_select" path passes only the
  // first CoreML device to the factory. That would test only one device, which CoreMLEP already covers, rather than
  // a selection containing both the NPU and GPU. Run only the V2 path, verify its input assignment directly, and
  // disable ORT CPU fallback so CoreML must execute the entire graph.
  RunBasicTest(kCoreMLExecutionProvider, std::nullopt, Ort::KeyValuePairs{}, select_devices,
               /*test_auto_select*/ false, AssertMul1SessionRunsOnCoreML, /*disable_cpu_ep_fallback*/ true);
}

// Verifies that AppendExecutionProvider_V2 passes both selected devices to the factory. CPUAndGPU is accepted only
// when the GPU is present in the factory's device list, and CPUAndNeuralEngine only when the NPU is present.
// Both sessions must be created successfully, so the test fails if either device is missing.
TEST(AutoEpSelection, CoreMLEPMultipleDevicesBothReachTheFactory) {
  const HardwareDeviceTypes hardware = GetHardwareDeviceTypes();
  if (!CoreMLCanClaimNpu(hardware) || !CoreMLCanClaimGpu(hardware)) {
    GTEST_SKIP() << "Need both an NPU and a GPU that the CoreML EP can advertise on this machine.";
  }

  const CoreMLDevices coreml_devices = GetCoreMLEpDevices();
  ASSERT_NE(coreml_devices.npu, nullptr) << "CoreML EP did not advertise the NPU hardware device.";
  ASSERT_NE(coreml_devices.gpu, nullptr) << "CoreML EP did not advertise the GPU hardware device.";

  std::vector<Ort::ConstEpDevice> ep_devices;
  ep_devices.emplace_back(coreml_devices.npu);
  ep_devices.emplace_back(coreml_devices.gpu);

  for (const char* compute_units : {kCoremlProviderOption_MLComputeUnits_CPUAndGPU,
                                    kCoremlProviderOption_MLComputeUnits_CPUAndNeuralEngine}) {
    SCOPED_TRACE(compute_units);

    Ort::KeyValuePairs ep_options;
    ep_options.Add(kCoremlProviderOption_MLComputeUnits, compute_units);

    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_V2(*ort_env, ep_devices, ep_options);

    // AppendExecutionProvider_V2 adds both devices and the current MLComputeUnits value to session_options.
    // Session creation must succeed because MLComputeUnits enables an accelerator included in the selection.
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
    ASSERT_NO_FATAL_FAILURE(AssertMul1SessionRunsOnCoreML(session));
  }
}

// Select only the CoreML GPU device explicitly via AppendExecutionProvider_V2 (the CPUAndGPU default path).
TEST(AutoEpSelection, CoreMLEPExplicitGpu) {
  if (!CoreMLCanClaimGpu(GetHardwareDeviceTypes())) {
    GTEST_SKIP() << "No GPU hardware device that the CoreML EP factory can advertise on this machine.";
  }

  const CoreMLDevices coreml_devices = GetCoreMLEpDevices();
  ASSERT_NE(coreml_devices.gpu, nullptr) << "CoreML EP did not claim the GPU hardware device.";

  const auto select_devices = [&](std::vector<const OrtEpDevice*>& devices) {
    devices = {coreml_devices.gpu};
  };

  // select_devices affects only the AppendExecutionProvider_V2 path. The "test.ep_to_select" path selects
  // only the first CoreML device, so it cannot guarantee a GPU-only selection. Verify that the session uses
  // CoreML, and disable ORT CPU fallback to prevent a false pass.
  RunBasicTest(kCoreMLExecutionProvider, std::nullopt, Ort::KeyValuePairs{}, select_devices,
               /*test_auto_select*/ false, AssertMul1SessionRunsOnCoreML, /*disable_cpu_ep_fallback*/ true);
}

// PREFER_NPU must select the CoreML EP for the Apple Neural Engine instead of falling back to the ORT CPU EP.
// MAX_EFFICIENCY and MIN_OVERALL_POWER use the same selector as PREFER_NPU, so the test verifies the same result
// for both aliases.
TEST(AutoEpSelection, CoreMLEPPreferNpu) {
  if (!CoreMLCanClaimNpu(GetHardwareDeviceTypes())) {
    GTEST_SKIP() << "No NPU hardware device, or this Core ML version cannot advertise it.";
  }
  ASSERT_NE(GetCoreMLEpDevices().npu, nullptr) << "CoreML EP did not claim the NPU hardware device.";

  for (const OrtExecutionProviderDevicePolicy policy : {OrtExecutionProviderDevicePolicy_PREFER_NPU,
                                                        OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY,
                                                        OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER}) {
    SCOPED_TRACE(testing::Message() << "policy " << policy);
    Ort::SessionOptions session_options;
    session_options.SetEpSelectionPolicy(policy);
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);

    ASSERT_NO_FATAL_FAILURE(AssertMul1SessionRunsOnCoreML(session));
  }
}

// PREFER_GPU must select the CoreML EP. When the internal WebGPU EP is also registered, both EPs advertise an
// OrtEpDevice for the same Apple GPU, and the test verifies that CoreML is selected over WebGPU.
// Without the internal WebGPU EP, the test simply verifies that PREFER_GPU selects CoreML.
// MAX_PERFORMANCE uses the same selector and is covered by the same assertions.
TEST(AutoEpSelection, CoreMLEPPreferGpu) {
  if (!CoreMLCanClaimGpu(GetHardwareDeviceTypes())) {
    GTEST_SKIP() << "No GPU hardware device that the CoreML EP factory can advertise on this machine.";
  }

  const CoreMLDevices coreml_devices = GetCoreMLEpDevices();
  ASSERT_NE(coreml_devices.gpu, nullptr) << "CoreML EP did not claim the GPU hardware device.";

#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
  // Verify that WebGPU also advertises an OrtEpDevice for the same GPU. Otherwise, selecting CoreML would not show
  // that it is preferred over WebGPU for the same hardware device.
  bool matching_webgpu_device_present = false;
  {
    const OrtApi* c_api = &Ort::GetApi();
    const OrtHardwareDevice* coreml_gpu_hw = c_api->EpDevice_Device(coreml_devices.gpu);
    const OrtEpDevice* const* ep_devices = nullptr;
    size_t num_devices = 0;

    ASSERT_ORTSTATUS_OK(c_api->GetEpDevices(*ort_env, &ep_devices, &num_devices));
    for (size_t i = 0; i < num_devices; ++i) {
      if (strcmp(c_api->EpDevice_EpName(ep_devices[i]), kWebGpuExecutionProvider) == 0 &&
          c_api->EpDevice_Device(ep_devices[i]) == coreml_gpu_hw) {
        matching_webgpu_device_present = true;
        break;
      }
    }
  }

  ASSERT_TRUE(matching_webgpu_device_present)
      << "Expected WebGPU to also advertise an OrtEpDevice for the CoreML GPU's hardware device in this build.";
#endif

  for (const OrtExecutionProviderDevicePolicy policy : {OrtExecutionProviderDevicePolicy_PREFER_GPU,
                                                        OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE}) {
    SCOPED_TRACE(testing::Message() << "policy " << policy);
    Ort::SessionOptions session_options;
    session_options.SetEpSelectionPolicy(policy);
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);

    ASSERT_NO_FATAL_FAILURE(AssertMul1SessionRunsOnCoreML(session));
  }
}

// A session created through the legacy AppendExecutionProvider API has no selected OrtEpDevice.
// SessionGetEpDeviceForInputs therefore matches the assigned EP name against the OrtEpDevices returned by
// GetEpDevices and reports the first match. Before CoreML was exposed through OrtEpFactory, GetEpDevices returned
// no CoreML OrtEpDevices, so this lookup found no match and the API reported nullptr. With the factory registered,
// the lookup reports the first advertised CoreML device, which may be an NPU or a GPU.
// Because the lookup matches only the EP name, the reported device does not reflect the session's MLComputeUnits.
// A CPUOnly session therefore reports the first advertised CoreML device. With the current factory, that device is
// an NPU or a GPU. Advertising a CPU device would not resolve this mismatch because the lookup would still return
// the first CoreML device in GetEpDevices, regardless of MLComputeUnits. This test captures the current behavior,
// not a final API contract. It remains undecided which device, if any, a legacy CoreML session should report.
TEST(AutoEpSelection, CoreMLLegacySessionReportsFirstAdvertisedEpDevice) {
  const OrtEpDevice* first_coreml_device = GetFirstCoreMLEpDevice();
  if (first_coreml_device == nullptr) {
    GTEST_SKIP() << "CoreML EP advertised no OrtEpDevice on this machine, so the lookup has nothing to report.";
  }

  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider(
      kCoreMLExecutionProvider,
      {{kCoremlProviderOption_MLComputeUnits, kCoremlProviderOption_MLComputeUnits_CPUOnly}});
  Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);

  const OrtApi* c_api = &Ort::GetApi();
  const OrtEpDevice* input_ep_device = nullptr;
  ASSERT_ORTSTATUS_OK(c_api->SessionGetEpDeviceForInputs(session, &input_ep_device, 1));

  ASSERT_NE(input_ep_device, nullptr) << "Expected the legacy CoreML session to report an OrtEpDevice.";
  EXPECT_STREQ(c_api->EpDevice_EpName(input_ep_device), kCoreMLExecutionProvider);
  EXPECT_EQ(input_ep_device, first_coreml_device)
      << "The legacy session has no selected OrtEpDevice, so the current name-based lookup should report the first "
         "advertised CoreML device.";

  // SessionGetEpDeviceForOutputs performs its own device lookup, so check it separately from
  // SessionGetEpDeviceForInputs.
  const OrtEpDevice* output_ep_device = nullptr;
  ASSERT_ORTSTATUS_OK(c_api->SessionGetEpDeviceForOutputs(session, &output_ep_device, 1));

  ASSERT_NE(output_ep_device, nullptr)
      << "Expected SessionGetEpDeviceForOutputs to report an OrtEpDevice for the legacy CoreML session.";
  EXPECT_STREQ(c_api->EpDevice_EpName(output_ep_device), kCoreMLExecutionProvider);
  EXPECT_EQ(output_ep_device, first_coreml_device)
      << "SessionGetEpDeviceForOutputs must report the same first advertised CoreML device as "
         "SessionGetEpDeviceForInputs.";
}

// Verifies how the factory sets MLComputeUnits when the caller does not specify it: CPUAndNeuralEngine for NPU,
// CPUAndGPU for GPU, and ALL for NPU plus GPU. Also verifies that a compatible user-provided value is preserved.
// The test uses synthetic hardware devices, so it also runs on Apple systems where discovery reports no NPU or
// GPU, such as Intel Macs.
TEST(AutoEpSelection, CoreMLEPDefaultComputeUnits) {
  OrtHardwareDevice npu{};
  npu.type = OrtHardwareDeviceType_NPU;
  OrtHardwareDevice gpu{};
  gpu.type = OrtHardwareDeviceType_GPU;

  const auto default_compute_units = [](std::initializer_list<const OrtHardwareDevice*> devices) -> std::string {
    ProviderOptions options;
    Ort::Status status{CoreMLEpFactory::ValidateDeviceSelectionAndResolveMLComputeUnits(
        devices.begin(), devices.size(), options)};
    if (!status.IsOK()) {
      ADD_FAILURE() << status.GetErrorMessage();
      return {};
    }

    const auto it = options.find(kCoremlProviderOption_MLComputeUnits);
    if (it == options.end()) {
      ADD_FAILURE() << "MLComputeUnits was not set.";
      return {};
    }

    return it->second;
  };

  EXPECT_EQ(default_compute_units({&npu}), "CPUAndNeuralEngine");
  EXPECT_EQ(default_compute_units({&gpu}), "CPUAndGPU");
  EXPECT_EQ(default_compute_units({&npu, &gpu}), "ALL");

  // A compatible user-provided MLComputeUnits value is preserved.
  ProviderOptions user_options{{kCoremlProviderOption_MLComputeUnits, "CPUOnly"}};
  const OrtHardwareDevice* npu_only[] = {&npu};
  Ort::Status status{
      CoreMLEpFactory::ValidateDeviceSelectionAndResolveMLComputeUnits(npu_only, 1, user_options)};
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  EXPECT_EQ(user_options[kCoremlProviderOption_MLComputeUnits], "CPUOnly");
}

// Verifies that MLComputeUnits reaches the factory through SessionOptions. With PREFER_NPU,
// MLComputeUnits=CPUAndGPU must be rejected because the policy did not select the GPU.
TEST(AutoEpSelection, CoreMLEPPreferNpuConflictingComputeUnits) {
  if (!CoreMLCanClaimNpu(GetHardwareDeviceTypes())) {
    GTEST_SKIP() << "No NPU hardware device, or this Core ML version cannot advertise it.";
  }
  ASSERT_NE(GetCoreMLEpDevices().npu, nullptr) << "CoreML EP did not claim the NPU hardware device.";

  Ort::SessionOptions session_options;
  session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_NPU);

  const std::string option_key =
      OrtSessionOptions::GetProviderOptionPrefix(kCoreMLExecutionProvider) + kCoremlProviderOption_MLComputeUnits;
  session_options.AddConfigEntry(option_key.c_str(), "CPUAndGPU");

  try {
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
    FAIL() << "Session creation should have failed: MLComputeUnits=CPUAndGPU conflicts with the NPU selection.";
  } catch (const Ort::Exception& ex) {
    EXPECT_EQ(ex.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
    EXPECT_THAT(ex.what(), ::testing::HasSubstr("was not selected"));
  }
}

// Verifies that AppendExecutionProvider_V2 passes ep_options to factory validation. With an explicit GPU-only
// selection, MLComputeUnits=CPUAndNeuralEngine must be rejected because the NPU was not selected.
TEST(AutoEpSelection, CoreMLEPExplicitGpuConflictingComputeUnits) {
  if (!CoreMLCanClaimGpu(GetHardwareDeviceTypes())) {
    GTEST_SKIP() << "No GPU hardware device that the CoreML EP factory can advertise on this machine.";
  }

  const CoreMLDevices coreml_devices = GetCoreMLEpDevices();
  ASSERT_NE(coreml_devices.gpu, nullptr) << "CoreML EP did not claim the GPU hardware device.";

  Ort::SessionOptions session_options;
  Ort::KeyValuePairs ep_options;
  ep_options.Add(kCoremlProviderOption_MLComputeUnits, "CPUAndNeuralEngine");
  std::vector<Ort::ConstEpDevice> ep_devices{Ort::ConstEpDevice{coreml_devices.gpu}};
  session_options.AppendExecutionProvider_V2(*ort_env, ep_devices, ep_options);

  try {
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
    FAIL() << "Session creation should have failed: MLComputeUnits=CPUAndNeuralEngine conflicts with the GPU "
              "selection.";
  } catch (const Ort::Exception& ex) {
    // Verify the diagnostic message independently of the V2 error-code handling.
    // CoreMLEPPreferNpuConflictingComputeUnits verifies ORT_INVALID_ARGUMENT through the policy path.
    EXPECT_THAT(ex.what(), ::testing::HasSubstr("was not selected"));
  }
}

// Verifies that an unknown MLComputeUnits value reaches CoreMLOptions and is rejected during provider creation.
// CoreMLEPComputeUnitsOverrideValidation separately verifies that the factory passes the value through.
TEST(AutoEpSelection, CoreMLEPUnknownComputeUnitsRejectedByProvider) {
  const CoreMLDevices coreml_devices = GetCoreMLEpDevices();
  const OrtEpDevice* device = coreml_devices.npu != nullptr ? coreml_devices.npu : coreml_devices.gpu;
  if (device == nullptr) {
    GTEST_SKIP() << "No CoreML OrtEpDevice on this machine.";
  }

  Ort::KeyValuePairs ep_options;
  ep_options.Add(kCoremlProviderOption_MLComputeUnits, "NotAComputeUnitsValue");
  std::vector<Ort::ConstEpDevice> ep_devices{Ort::ConstEpDevice{device}};

  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_V2(*ort_env, ep_devices, ep_options);

  try {
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
    FAIL() << "Session creation should have failed: MLComputeUnits=NotAComputeUnitsValue is not a valid value.";
  } catch (const Ort::Exception& ex) {
    EXPECT_THAT(ex.what(), ::testing::HasSubstr("Invalid value for option"));
    EXPECT_THAT(ex.what(), ::testing::HasSubstr("NotAComputeUnitsValue"));
  }
}

// Verifies that MLComputeUnits=CPUOnly can narrow an accelerator selection. The session must be created and run
// successfully, and the graph must remain assigned to the CoreML EP. CoreML may enable only accelerators represented
// by the selected OrtEpDevices. This does not prevent Core ML from using its internal CPU path.
// MLComputeUnits=CPUOnly also does not reassign the graph to the ORT CPU EP.
TEST(AutoEpSelection, CoreMLEPNarrowingComputeUnitsOverride) {
  const HardwareDeviceTypes hardware = GetHardwareDeviceTypes();
  const bool can_claim_npu = CoreMLCanClaimNpu(hardware);
  const bool can_claim_gpu = CoreMLCanClaimGpu(hardware);
  if (!can_claim_npu && !can_claim_gpu) {
    GTEST_SKIP() << "No hardware device that the CoreML EP factory can advertise on this machine.";
  }

  const CoreMLDevices coreml_devices = GetCoreMLEpDevices();
  std::vector<Ort::ConstEpDevice> ep_devices;
  if (can_claim_npu) {
    ASSERT_NE(coreml_devices.npu, nullptr) << "CoreML EP did not claim the NPU hardware device.";
    ep_devices.emplace_back(coreml_devices.npu);
  }
  if (can_claim_gpu) {
    ASSERT_NE(coreml_devices.gpu, nullptr) << "CoreML EP did not claim the GPU hardware device.";
    ep_devices.emplace_back(coreml_devices.gpu);
  }

  Ort::KeyValuePairs ep_options;
  ep_options.Add(kCoremlProviderOption_MLComputeUnits, "CPUOnly");

  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_V2(*ort_env, ep_devices, ep_options);
  Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);

  ASSERT_NO_FATAL_FAILURE(AssertMul1SessionRunsOnCoreML(session));
}

// Verifies that a user-provided MLComputeUnits value is compatible with the selected accelerators. The value
// may disable selected accelerators, as CPUOnly does, but it must not enable an accelerator that was not selected.
TEST(AutoEpSelection, CoreMLEPComputeUnitsOverrideValidation) {
  OrtHardwareDevice npu{};
  npu.type = OrtHardwareDeviceType_NPU;
  OrtHardwareDevice gpu{};
  gpu.type = OrtHardwareDeviceType_GPU;

  const auto validate = [](std::initializer_list<const OrtHardwareDevice*> devices,
                           std::string_view compute_units) {
    ProviderOptions options{{kCoremlProviderOption_MLComputeUnits, std::string{compute_units}}};
    return Ort::Status{CoreMLEpFactory::ValidateDeviceSelectionAndResolveMLComputeUnits(
        devices.begin(), devices.size(), options)};
  };

  // Values that enable only selected accelerators, or disable them with CPUOnly, are allowed.
  EXPECT_TRUE(validate({&npu}, "CPUAndNeuralEngine").IsOK());
  EXPECT_TRUE(validate({&npu}, "CPUOnly").IsOK());
  EXPECT_TRUE(validate({&gpu}, "CPUAndGPU").IsOK());
  EXPECT_TRUE(validate({&gpu}, "CPUOnly").IsOK());
  EXPECT_TRUE(validate({&npu, &gpu}, "ALL").IsOK());
  EXPECT_TRUE(validate({&npu, &gpu}, "CPUAndNeuralEngine").IsOK());
  EXPECT_TRUE(validate({&npu, &gpu}, "CPUAndGPU").IsOK());
  EXPECT_TRUE(validate({&npu, &gpu}, "CPUOnly").IsOK());

  // Values that enable an unselected accelerator are rejected.
  ExpectInvalidArgument(validate({&npu}, "CPUAndGPU"), "NPU selected, CPUAndGPU requested");
  ExpectInvalidArgument(validate({&npu}, "ALL"), "NPU selected, ALL requested");
  ExpectInvalidArgument(validate({&gpu}, "CPUAndNeuralEngine"), "GPU selected, CPUAndNeuralEngine requested");
  ExpectInvalidArgument(validate({&gpu}, "ALL"), "GPU selected, ALL requested");

  // This factory helper checks only whether MLComputeUnits is compatible with the selected devices.
  // Unknown values are left for CoreMLOptions to reject during provider creation.
  EXPECT_TRUE(validate({&npu}, "NotAComputeUnitsValue").IsOK());
}

// Tests which device selections the factory accepts or rejects.
// Valid selections contain one NPU, one GPU, or one NPU plus one GPU.
// Empty lists, CPU devices, duplicate device types, a null device entry, and a null device-list pointer with a
// nonzero device count must return ORT_INVALID_ARGUMENT.
TEST(AutoEpSelection, CoreMLEPDeviceValidation) {
  OrtHardwareDevice npu{};
  npu.type = OrtHardwareDeviceType_NPU;
  OrtHardwareDevice gpu{};
  gpu.type = OrtHardwareDeviceType_GPU;
  OrtHardwareDevice cpu{};
  cpu.type = OrtHardwareDeviceType_CPU;

  const auto validate = [](std::initializer_list<const OrtHardwareDevice*> devices) {
    ProviderOptions options;
    return Ort::Status{CoreMLEpFactory::ValidateDeviceSelectionAndResolveMLComputeUnits(
        devices.begin(), devices.size(), options)};
  };

  // The valid selections, for contrast.
  EXPECT_TRUE(validate({&npu}).IsOK());
  EXPECT_TRUE(validate({&gpu}).IsOK());
  EXPECT_TRUE(validate({&npu, &gpu}).IsOK());

  ExpectInvalidArgument(validate({}), "no devices");
  ExpectInvalidArgument(validate({&cpu}), "CPU only: the factory never claims the CPU device");
  ExpectInvalidArgument(validate({&cpu, &gpu}), "GPU mixed with an unsupported device");
  ExpectInvalidArgument(validate({&npu, &npu}), "duplicate NPU");
  ExpectInvalidArgument(validate({&gpu, &gpu}), "duplicate GPU");

  ExpectInvalidArgument(validate({nullptr}), "null device");
  ProviderOptions options;
  const Ort::Status null_device_list_status{
      CoreMLEpFactory::ValidateDeviceSelectionAndResolveMLComputeUnits(nullptr, 1, options)};
  ExpectInvalidArgument(null_device_list_status, "null device list");
}

// Verifies CanClaimDeviceType independently of the host hardware and runtime Core ML version.
// This test covers the Core ML 5 minimum, the Core ML 6 NPU requirement, and the version-independent CPU exclusion.
TEST(AutoEpSelection, CoreMLEPClaimabilityByVersion) {
  const auto can_claim = [](OrtHardwareDeviceType type, int32_t coreml_version) {
    return CoreMLEpFactory::CanClaimDeviceType(type, coreml_version);
  };

  // Below the EP's Core ML 5 minimum (provider creation would fail) nothing is advertised.
  EXPECT_FALSE(can_claim(OrtHardwareDeviceType_NPU, 4));
  EXPECT_FALSE(can_claim(OrtHardwareDeviceType_GPU, 4));

  // Core ML 5: the GPU is advertised, the NPU is not (CPUAndNeuralEngine requires Core ML 6).
  EXPECT_FALSE(can_claim(OrtHardwareDeviceType_NPU, 5));
  EXPECT_TRUE(can_claim(OrtHardwareDeviceType_GPU, 5));

  // Core ML 6+: both.
  EXPECT_TRUE(can_claim(OrtHardwareDeviceType_NPU, 6));
  EXPECT_TRUE(can_claim(OrtHardwareDeviceType_GPU, 6));
  EXPECT_TRUE(can_claim(OrtHardwareDeviceType_NPU, 7));
  EXPECT_TRUE(can_claim(OrtHardwareDeviceType_GPU, 7));

  // The CPU device is never claimed regardless of version (left to the CPU EP).
  EXPECT_FALSE(can_claim(OrtHardwareDeviceType_CPU, 4));
  EXPECT_FALSE(can_claim(OrtHardwareDeviceType_CPU, 8));
}

// Tests SelectDevicesToClaim with synthetic hardware devices instead of relying on host discovery. Covers CPU
// filtering, input order, duplicate devices of the same type, the max_ep_devices limit, and version checks
// independently of the Core ML version installed on the host.
TEST(AutoEpSelection, CoreMLEPSelectDevicesToClaim) {
  OrtHardwareDevice npu0{};
  npu0.type = OrtHardwareDeviceType_NPU;
  OrtHardwareDevice npu1{};
  npu1.type = OrtHardwareDeviceType_NPU;
  OrtHardwareDevice gpu0{};
  gpu0.type = OrtHardwareDeviceType_GPU;
  OrtHardwareDevice gpu1{};
  gpu1.type = OrtHardwareDeviceType_GPU;
  OrtHardwareDevice cpu{};
  cpu.type = OrtHardwareDeviceType_CPU;

  constexpr int32_t coreml6 = 6;
  const auto select = [](std::initializer_list<const OrtHardwareDevice*> devices, int32_t coreml_version,
                         size_t max_ep_devices = 8) {
    return CoreMLEpFactory::SelectDevicesToClaim(gsl::make_span(devices.begin(), devices.size()), coreml_version,
                                                 max_ep_devices);
  };

  // CPU + NPU + GPU -> the CPU is filtered out. The NPU and GPU remain in input order.
  {
    const auto selected = select({&cpu, &npu0, &gpu0}, coreml6);
    ASSERT_EQ(selected.size(), 2u);
    EXPECT_EQ(selected[0], &npu0);
    EXPECT_EQ(selected[1], &gpu0);
  }

  // Repeat the NPU+GPU selection with the GPU first. Together with the NPU-first case above, this verifies that
  // selection preserves input order instead of sorting by device type.
  {
    const auto selected = select({&cpu, &gpu0, &npu0}, coreml6);
    ASSERT_EQ(selected.size(), 2u);
    EXPECT_EQ(selected[0], &gpu0);
    EXPECT_EQ(selected[1], &npu0);
  }

  // Duplicate devices of a type: only the first of each type is advertised.
  {
    const auto selected = select({&gpu0, &gpu1}, coreml6);
    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0], &gpu0);
  }
  {
    const auto selected = select({&npu0, &npu1, &gpu0, &gpu1}, coreml6);
    ASSERT_EQ(selected.size(), 2u);
    EXPECT_EQ(selected[0], &npu0);
    EXPECT_EQ(selected[1], &gpu0);
  }

  // max_ep_devices limits the number of selected devices.
  {
    const auto selected = select({&npu0, &gpu0}, coreml6, /*max_ep_devices*/ 1);
    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0], &npu0);
  }

  // Verify that selection applies the CanClaimDeviceType version checks.
  {
    const auto selected = select({&npu0, &gpu0}, /*coreml_version*/ 5);
    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0], &gpu0);
  }
  EXPECT_TRUE(select({&npu0, &gpu0}, /*coreml_version*/ 4).empty());
}

// Tests CreateAndPublishEpDevices with injected create and release callbacks. On failure, all previously
// created devices must be released and the output parameters must remain unchanged. On success, all device
// pointers must be published without releasing any devices. The dummy OrtEpDevice objects are never dereferenced.
// Their addresses identify which devices are published or released.
TEST(AutoEpSelection, CoreMLEPCreateAndPublishEpDevices) {
  OrtHardwareDevice npu{};
  npu.type = OrtHardwareDeviceType_NPU;
  OrtHardwareDevice gpu{};
  gpu.type = OrtHardwareDeviceType_GPU;
  const std::array<const OrtHardwareDevice*, 2> selected{&npu, &gpu};

  OrtEpDevice dummy_device0{};
  OrtEpDevice dummy_device1{};

  InlinedVector<OrtEpDevice*, 2> released;
  const auto release = [&released](OrtEpDevice* ep_device) { released.push_back(ep_device); };

  // Initialize the caller-owned output array with distinct non-null pointers. Failure cases verify that these values
  // remain unchanged. Initializing the array with nullptr would not distinguish "left untouched" from
  // "cleared to nullptr".
  OrtEpDevice sentinel0{};
  OrtEpDevice sentinel1{};
  const std::array<OrtEpDevice*, 2> untouched{&sentinel0, &sentinel1};

  // The second device creation fails. The first device must be released, and the output parameters must remain
  // unchanged.
  {
    size_t create_calls = 0;
    const auto create_second_fails = [&](const OrtHardwareDevice&, OrtEpDevice** ep_device) -> OrtStatus* {
      if (create_calls++ == 0) {
        *ep_device = &dummy_device0;
        return nullptr;
      }

      return Ort::GetApi().CreateStatus(ORT_FAIL, "injected creation failure");
    };

    std::array<OrtEpDevice*, 2> ep_devices = untouched;
    size_t num_ep_devices = 42;  // Initial value that must remain unchanged on failure.

    Ort::Status status{CoreMLEpFactory::CreateAndPublishEpDevices(gsl::make_span(selected), create_second_fails,
                                                                  release, ep_devices.data(), &num_ep_devices)};
    ASSERT_FALSE(status.IsOK());
    EXPECT_THAT(status.GetErrorMessage(), ::testing::HasSubstr("injected creation failure"));
    EXPECT_THAT(released, ::testing::ElementsAre(&dummy_device0));
    EXPECT_THAT(ep_devices, ::testing::ElementsAreArray(untouched));
    EXPECT_EQ(num_ep_devices, size_t{42});
  }

  // The first device creation fails. No devices need to be released, and the output parameters must remain unchanged.
  {
    released.clear();
    const auto create_first_fails = [](const OrtHardwareDevice&, OrtEpDevice**) -> OrtStatus* {
      return Ort::GetApi().CreateStatus(ORT_FAIL, "injected creation failure");
    };

    std::array<OrtEpDevice*, 2> ep_devices = untouched;
    size_t num_ep_devices = 42;  // Initial value that must remain unchanged on failure.

    Ort::Status status{CoreMLEpFactory::CreateAndPublishEpDevices(gsl::make_span(selected), create_first_fails,
                                                                  release, ep_devices.data(), &num_ep_devices)};
    ASSERT_FALSE(status.IsOK());
    EXPECT_TRUE(released.empty());
    EXPECT_THAT(ep_devices, ::testing::ElementsAreArray(untouched));
    EXPECT_EQ(num_ep_devices, size_t{42});
  }

  // Success: both devices are published, nothing is released.
  {
    released.clear();
    size_t create_calls = 0;
    const auto create_ok = [&](const OrtHardwareDevice&, OrtEpDevice** ep_device) -> OrtStatus* {
      *ep_device = (create_calls++ == 0) ? &dummy_device0 : &dummy_device1;
      return nullptr;
    };

    std::array<OrtEpDevice*, 2> ep_devices{nullptr, nullptr};
    size_t num_ep_devices = 0;

    Ort::Status status{CoreMLEpFactory::CreateAndPublishEpDevices(gsl::make_span(selected), create_ok, release,
                                                                  ep_devices.data(), &num_ep_devices)};
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    EXPECT_EQ(num_ep_devices, size_t{2});
    EXPECT_EQ(ep_devices[0], &dummy_device0);
    EXPECT_EQ(ep_devices[1], &dummy_device1);
    EXPECT_TRUE(released.empty());
  }

  // No devices selected: a successful call still sets the output count to zero, as required by GetSupportedDevices.
  {
    released.clear();
    const auto create_never_called = [](const OrtHardwareDevice&, OrtEpDevice**) -> OrtStatus* {
      ADD_FAILURE() << "create_ep_device must not be called when nothing was selected.";
      return Ort::GetApi().CreateStatus(ORT_FAIL, "unexpected call");
    };

    std::array<OrtEpDevice*, 2> ep_devices = untouched;
    size_t num_ep_devices = 42;

    Ort::Status status{CoreMLEpFactory::CreateAndPublishEpDevices(gsl::span<const OrtHardwareDevice* const>{},
                                                                  create_never_called, release, ep_devices.data(),
                                                                  &num_ep_devices)};
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    EXPECT_EQ(num_ep_devices, size_t{0});
    EXPECT_THAT(ep_devices, ::testing::ElementsAreArray(untouched));
    EXPECT_TRUE(released.empty());
  }
}
#endif  // defined(USE_COREML) && defined(__APPLE__)

#if defined(USE_COREML) && !defined(__APPLE__)
// CoreML can be built on non-Apple platforms to test model conversion, but it cannot execute models there.
// The factory checks only the device type and Core ML version. Because the non-Apple CoreMLVersion stub reports
// Core ML 7, the factory would accept accelerators discovered on a non-Apple system. Verify that the platform guards
// prevent the CoreML factory from being registered on non-Apple builds.
// Registration is checked with GetHardwareDeviceEpIncompatibilityDetails rather than through the published device
// list. The function looks up the factory by EP name before checking hardware compatibility, so any discovered
// device, including the CPU, is sufficient and the test does not depend on accelerator discovery.
TEST(AutoEpSelection, CoreMLEPIsNotRegisteredOnNonApplePlatforms) {
  const OrtApi* c_api = &Ort::GetApi();

  size_t num_devices = 0;
  ASSERT_ORTSTATUS_OK(c_api->GetNumHardwareDevices(*ort_env, &num_devices));
  ASSERT_GT(num_devices, 0u) << "Expected device discovery to report at least the CPU device.";

  std::vector<const OrtHardwareDevice*> devices(num_devices);
  ASSERT_ORTSTATUS_OK(c_api->GetHardwareDevices(*ort_env, devices.data(), num_devices));

  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  Ort::Status status{c_api->GetHardwareDeviceEpIncompatibilityDetails(*ort_env, kCoreMLExecutionProvider,
                                                                      devices.front(), &details)};
  // If the CoreML factory is unexpectedly registered, the call succeeds and returns an object that must be released
  // before the test reports the failure.
  if (details != nullptr) {
    c_api->ReleaseDeviceEpIncompatibilityDetails(details);
  }

  ASSERT_FALSE(status.IsOK()) << "A factory named " << kCoreMLExecutionProvider << " is registered on a non-Apple "
                              << "platform, where CoreML cannot execute models.";
  EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
  // ORT_INVALID_ARGUMENT alone does not prove that the factory is absent. If the EP name matches a registered
  // factory, GetHardwareDeviceEpIncompatibilityDetails invokes the factory's compatibility callback,
  // which may return the same error code. Verify the diagnostic to confirm that no factory matched the CoreML EP name.
  EXPECT_THAT(status.GetErrorMessage(), ::testing::HasSubstr("No valid factory found for execution provider"));
}
#endif  // defined(USE_COREML) && !defined(__APPLE__)

// tests for AutoEP selection related things in the API that aren't covered by the other tests.
TEST(AutoEpSelection, MiscApiTests) {
  const OrtApi* c_api = &Ort::GetApi();

  // nullptr and empty input to OrtKeyValuePairs. also test RemoveKeyValuePair
  {
    OrtKeyValuePairs* kvps = nullptr;
    c_api->CreateKeyValuePairs(&kvps);
    c_api->AddKeyValuePair(kvps, "key1", nullptr);    // should be ignored
    c_api->AddKeyValuePair(kvps, nullptr, "value1");  // should be ignored
    c_api->RemoveKeyValuePair(kvps, nullptr);         // should be ignored
    c_api->AddKeyValuePair(kvps, "", "value2");       // should be ignored
    ASSERT_EQ(c_api->GetKeyValue(kvps, ""), nullptr);

    c_api->AddKeyValuePair(kvps, "key1", "value1");
    c_api->AddKeyValuePair(kvps, "key2", "");  // empty value is allowed
    ASSERT_EQ(c_api->GetKeyValue(kvps, "key2"), std::string(""));

    c_api->RemoveKeyValuePair(kvps, "key1");
    const char* const* keys = nullptr;
    const char* const* values = nullptr;
    size_t num_entries = 0;
    c_api->GetKeyValuePairs(kvps, &keys, &values, &num_entries);
    ASSERT_EQ(num_entries, 1);

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

TEST(AutoEpSelection, PreferCpu) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  const Ort::KeyValuePairs provider_options;

  TestInference<float>(*ort_env, ORT_TSTR("testdata/mul_1.onnx"),
                       "",  // don't need EP name
                       std::nullopt,
                       provider_options,
                       inputs,
                       "Y",
                       expected_dims_y,
                       expected_values_y,
                       /* auto_select */ true,
                       /*select_devices*/ nullptr,
                       OrtExecutionProviderDevicePolicy::OrtExecutionProviderDevicePolicy_PREFER_CPU);
}

// this should fallback to CPU if no GPU
TEST(AutoEpSelection, PreferGpu) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  const Ort::KeyValuePairs provider_options;

  TestInference<float>(*ort_env, ORT_TSTR("testdata/mul_1.onnx"),
                       "",  // don't need EP name
                       std::nullopt,
                       provider_options,
                       inputs,
                       "Y",
                       expected_dims_y,
                       expected_values_y,
                       /* auto_select */ true,
                       /*select_devices*/ nullptr,
                       OrtExecutionProviderDevicePolicy::OrtExecutionProviderDevicePolicy_PREFER_GPU);
}

// this should fallback to CPU if no NPU
TEST(AutoEpSelection, PreferNpu) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  const Ort::KeyValuePairs provider_options;

  TestInference<float>(*ort_env, ORT_TSTR("testdata/mul_1.onnx"),
                       "",  // don't need EP name
                       std::nullopt,
                       provider_options,
                       inputs,
                       "Y",
                       expected_dims_y,
                       expected_values_y,
                       /* auto_select */ true,
                       /*select_devices*/ nullptr,
                       OrtExecutionProviderDevicePolicy::OrtExecutionProviderDevicePolicy_PREFER_NPU);
}

static OrtStatus* ORT_API_CALL PolicyDelegate(_In_ const OrtEpDevice** ep_devices,
                                              _In_ size_t num_devices,
                                              _In_ const OrtKeyValuePairs* model_metadata,
                                              _In_opt_ const OrtKeyValuePairs* /*runtime_metadata*/,
                                              _Inout_ const OrtEpDevice** selected,
                                              _In_ size_t max_selected,
                                              _Out_ size_t* num_selected,
                                              _In_ void* /*state*/) {
  *num_selected = 0;

  if (max_selected <= 2) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "Expected to be able to select 2 devices.");
  }

  if (model_metadata->Entries().empty()) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "Model metadata was empty.");
  }

  selected[0] = ep_devices[0];
  *num_selected = 1;
  if (num_devices > 1) {
    // CPU EP is always last.
    selected[1] = ep_devices[num_devices - 1];
    *num_selected = 2;
  }

  return nullptr;
}

static OrtStatus* ORT_API_CALL PolicyDelegateSelectNone(_In_ const OrtEpDevice** /*ep_devices*/,
                                                        _In_ size_t /*num_devices*/,
                                                        _In_ const OrtKeyValuePairs* /*model_metadata*/,
                                                        _In_opt_ const OrtKeyValuePairs* /*runtime_metadata*/,
                                                        _Inout_ const OrtEpDevice** /*selected*/,
                                                        _In_ size_t /*max_selected*/,
                                                        _Out_ size_t* num_selected,
                                                        _In_ void* /*state*/) {
  *num_selected = 0;

  return nullptr;
}

static OrtStatus* ORT_API_CALL PolicyDelegateReturnError(_In_ const OrtEpDevice** /*ep_devices*/,
                                                         _In_ size_t /*num_devices*/,
                                                         _In_ const OrtKeyValuePairs* /*model_metadata*/,
                                                         _In_opt_ const OrtKeyValuePairs* /*runtime_metadata*/,
                                                         _Inout_ const OrtEpDevice** /*selected*/,
                                                         _In_ size_t /*max_selected*/,
                                                         _Out_ size_t* num_selected,
                                                         _In_ void* /*state*/) {
  *num_selected = 0;

  return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT, "Selection error.");
}

// test providing a delegate
TEST(AutoEpSelection, PolicyDelegate) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  const Ort::KeyValuePairs provider_options;

  TestInference<float>(*ort_env, ORT_TSTR("testdata/mul_1.onnx"),
                       "",  // don't need EP name
                       std::nullopt,
                       provider_options,
                       inputs,
                       "Y",
                       expected_dims_y,
                       expected_values_y,
                       /* auto_select */ true,
                       /*select_devices*/ nullptr,
                       std::nullopt,
                       PolicyDelegate);
}

// test providing a delegate
TEST(AutoEpSelection, PolicyDelegateSelectsNothing) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  const Ort::KeyValuePairs provider_options;

  ASSERT_THROW(
      TestInference<float>(*ort_env, ORT_TSTR("testdata/mul_1.onnx"),
                           "",  // don't need EP name
                           std::nullopt,
                           provider_options,
                           inputs,
                           "Y",
                           expected_dims_y,
                           expected_values_y,
                           /* auto_select */ true,
                           /*select_devices*/ nullptr,
                           std::nullopt,
                           PolicyDelegateSelectNone,
                           /*test_session_creation_only*/ true),
      Ort::Exception);
}

TEST(AutoEpSelection, PolicyDelegateReturnsError) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  const Ort::KeyValuePairs provider_options;

  ASSERT_THROW(
      TestInference<float>(*ort_env, ORT_TSTR("testdata/mul_1.onnx"),
                           "",  // don't need EP name
                           std::nullopt,
                           provider_options,
                           inputs,
                           "Y",
                           expected_dims_y,
                           expected_values_y,
                           /* auto_select */ true,
                           /*select_devices*/ nullptr,
                           std::nullopt,
                           PolicyDelegateReturnError,
                           /*test_session_creation_only*/ true),
      Ort::Exception);
}

}  // namespace test
}  // namespace onnxruntime
