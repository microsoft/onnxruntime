// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <filesystem>
// #include <absl/base/config.h>
#include <gmock/gmock.h>
#include <gsl/gsl>
#include <gtest/gtest.h>

#include "core/graph/constants.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#if defined(USE_COREML)
#include "core/common/inlined_containers.h"
#endif

#if defined(USE_COREML) && defined(__APPLE__)
#include "core/providers/coreml/coreml_provider_factory.h"
#include "core/providers/coreml/model/host_utils.h"
#include "core/session/plugin_ep/ep_factory_coreml.h"
#include "core/session/plugin_ep/ep_factory_internal.h"
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

// Returns whether this machine has an NPU that the CoreML EP factory may advertise at the runtime Core ML version.
bool CoreMLCanClaimNpu(const HardwareDeviceTypes& hardware) {
  return hardware.has_npu && coreml::util::CoreMLVersion() >= MINIMUM_COREML_VERSION_FOR_NEURAL_ENGINE_SELECTION;
}

// Returns whether this machine has a GPU that the CoreML EP factory may advertise at the runtime Core ML version.
bool CoreMLCanClaimGpu(const HardwareDeviceTypes& hardware) {
  return hardware.has_gpu && coreml::util::CoreMLVersion() >= MINIMUM_COREML_VERSION;
}

// Verifies that a session for mul_1.onnx assigns its input to the CoreML EP. Used as the session checker for
// RunBasicTest, which runs the inference itself afterwards.
void AssertMul1InputAssignedToCoreML(Ort::Session& session) {
  const OrtApi* c_api = &Ort::GetApi();
  const OrtEpDevice* input_ep_device = nullptr;
  ASSERT_ORTSTATUS_OK(c_api->SessionGetEpDeviceForInputs(session, &input_ep_device, 1));
  ASSERT_NE(input_ep_device, nullptr);
  ASSERT_STREQ(c_api->EpDevice_EpName(input_ep_device), kCoreMLExecutionProvider);
}

// Verifies the input assignment and runs the inference. Used by the tests that create the session themselves.
void AssertMul1SessionRunsOnCoreML(Ort::Session& session) {
  ASSERT_NO_FATAL_FAILURE(AssertMul1InputAssignedToCoreML(session));

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
               /*test_auto_select*/ true, AssertMul1InputAssignedToCoreML, /*disable_cpu_ep_fallback*/ true);
}

// Tests explicit device selection through AppendExecutionProvider_V2. Each test case selects one or more CoreML
// devices, optionally provides MLComputeUnits in ep_options, and verifies that the session assigns the graph to
// CoreML and runs successfully. Test cases are skipped individually only when a required device is unavailable.
// Test cases with MLComputeUnits verify that the caller may narrow accelerator use within the selected devices:
// for an NPU-and-GPU selection, CPUAndGPU enables only the GPU accelerator and CPUAndNeuralEngine enables only
// the NPU. CPUOnly disables all selected accelerators while the graph remains assigned to CoreML.
TEST(AutoEpSelection, CoreMLEPExplicitDeviceSelection) {
  enum class Selection { kNpuAndGpu,
                         kGpuOnly,
                         kNpuOnly,
                         kEveryAdvertisedDevice };

  struct TestCase {
    const char* description;
    Selection selection;
    const char* compute_units;  // nullptr: MLComputeUnits omitted
  };

  const TestCase test_cases[] = {
      {"NPU and GPU, MLComputeUnits omitted", Selection::kNpuAndGpu, nullptr},
      {"NPU and GPU, CPUAndGPU", Selection::kNpuAndGpu, "CPUAndGPU"},
      {"NPU and GPU, CPUAndNeuralEngine", Selection::kNpuAndGpu, "CPUAndNeuralEngine"},
      {"GPU only, MLComputeUnits omitted", Selection::kGpuOnly, nullptr},
      {"NPU only, MLComputeUnits omitted", Selection::kNpuOnly, nullptr},
      {"every advertised device, CPUOnly", Selection::kEveryAdvertisedDevice, "CPUOnly"},
  };

  const HardwareDeviceTypes hardware = GetHardwareDeviceTypes();
  const bool can_claim_npu = CoreMLCanClaimNpu(hardware);
  const bool can_claim_gpu = CoreMLCanClaimGpu(hardware);
  if (!can_claim_npu && !can_claim_gpu) {
    GTEST_SKIP() << "No hardware device that the CoreML EP factory can advertise on this machine.";
  }

  const CoreMLDevices coreml_devices = GetCoreMLEpDevices();
  if (can_claim_npu) {
    ASSERT_NE(coreml_devices.npu, nullptr) << "CoreML EP did not claim the NPU hardware device.";
  }
  if (can_claim_gpu) {
    ASSERT_NE(coreml_devices.gpu, nullptr) << "CoreML EP did not claim the GPU hardware device.";
  }

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.description);

    const bool wants_npu = test_case.selection == Selection::kNpuAndGpu ||
                           test_case.selection == Selection::kNpuOnly ||
                           (test_case.selection == Selection::kEveryAdvertisedDevice && can_claim_npu);
    const bool wants_gpu = test_case.selection == Selection::kNpuAndGpu ||
                           test_case.selection == Selection::kGpuOnly ||
                           (test_case.selection == Selection::kEveryAdvertisedDevice && can_claim_gpu);
    if ((wants_npu && !can_claim_npu) || (wants_gpu && !can_claim_gpu)) {
      continue;
    }

    std::vector<const OrtEpDevice*> devices;
    if (wants_npu) {
      devices.push_back(coreml_devices.npu);
    }
    if (wants_gpu) {
      devices.push_back(coreml_devices.gpu);
    }

    Ort::KeyValuePairs ep_options;
    if (test_case.compute_units != nullptr) {
      ep_options.Add(kCoremlProviderOption_MLComputeUnits, test_case.compute_units);
    }

    // Run only the AppendExecutionProvider_V2 path because it passes the test case's full device selection to the
    // factory. The "test.ep_to_select" path would pass only the first CoreML device. Verify the input assignment and
    // disable ORT CPU fallback so CoreML must execute the entire graph.
    RunBasicTest(
        kCoreMLExecutionProvider, std::nullopt, ep_options,
        [&devices](std::vector<const OrtEpDevice*>& selected) { selected = devices; },
        /*test_auto_select*/ false, AssertMul1InputAssignedToCoreML, /*disable_cpu_ep_fallback*/ true);
  }
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

// MLComputeUnits must not enable an accelerator outside the selected devices. The policy path uses PREFER_NPU to
// select only the NPU, then tests CPUAndGPU and ALL. The AppendExecutionProvider_V2 path selects only the GPU, then
// tests CPUAndNeuralEngine and ALL. Each value must be rejected because it enables an unselected accelerator.
TEST(AutoEpSelection, CoreMLEPConflictingComputeUnitsRejected) {
  const HardwareDeviceTypes hardware = GetHardwareDeviceTypes();
  const bool can_claim_npu = CoreMLCanClaimNpu(hardware);
  const bool can_claim_gpu = CoreMLCanClaimGpu(hardware);
  if (!can_claim_npu && !can_claim_gpu) {
    GTEST_SKIP() << "No hardware device that the CoreML EP factory can advertise on this machine.";
  }

  const CoreMLDevices coreml_devices = GetCoreMLEpDevices();

  const auto expect_rejected = [](Ort::SessionOptions& session_options, const char* explanation,
                                  bool check_error_code) {
    try {
      Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
      FAIL() << "Expected session creation to fail: " << explanation;
    } catch (const Ort::Exception& ex) {
      if (check_error_code) {
        EXPECT_EQ(ex.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
      }
      EXPECT_THAT(ex.what(), ::testing::HasSubstr("was not selected"));
    }
  };

  if (can_claim_npu) {
    ASSERT_NE(coreml_devices.npu, nullptr) << "CoreML EP did not claim the NPU hardware device.";

    for (const char* compute_units : {"CPUAndGPU", "ALL"}) {
      SCOPED_TRACE(compute_units);
      Ort::SessionOptions session_options;
      session_options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_NPU);
      const std::string option_key =
          OrtSessionOptions::GetProviderOptionPrefix(kCoreMLExecutionProvider) + kCoremlProviderOption_MLComputeUnits;
      session_options.AddConfigEntry(option_key.c_str(), compute_units);
      expect_rejected(session_options, "the value enables the GPU, which PREFER_NPU did not select.",
                      /*check_error_code*/ true);
    }
  }

  if (can_claim_gpu) {
    ASSERT_NE(coreml_devices.gpu, nullptr) << "CoreML EP did not claim the GPU hardware device.";

    for (const char* compute_units : {"CPUAndNeuralEngine", "ALL"}) {
      SCOPED_TRACE(compute_units);
      Ort::KeyValuePairs ep_options;
      ep_options.Add(kCoremlProviderOption_MLComputeUnits, compute_units);
      std::vector<Ort::ConstEpDevice> ep_devices{Ort::ConstEpDevice{coreml_devices.gpu}};
      Ort::SessionOptions session_options;
      session_options.AppendExecutionProvider_V2(*ort_env, ep_devices, ep_options);
      // The V2 registration path currently reports the factory's ORT_INVALID_ARGUMENT as ORT_FAIL, so only the
      // diagnostic message is verified here. The policy path above verifies ORT_INVALID_ARGUMENT.
      expect_rejected(session_options, "the value enables the NPU, but only the GPU was selected.",
                      /*check_error_code*/ false);
    }
  }
}

// The factory must reject selecting the same device type twice. AppendExecutionProvider_V2 validation
// checks only that all selected devices use the same EP and factory, so the duplicate reaches the factory-specific
// validation.
TEST(AutoEpSelection, CoreMLEPDuplicateDeviceSelectionRejected) {
  const HardwareDeviceTypes hardware = GetHardwareDeviceTypes();
  if (!CoreMLCanClaimNpu(hardware) && !CoreMLCanClaimGpu(hardware)) {
    GTEST_SKIP() << "No hardware device that the CoreML EP factory can advertise on this machine.";
  }

  const CoreMLDevices coreml_devices = GetCoreMLEpDevices();
  ASSERT_TRUE(coreml_devices.npu != nullptr || coreml_devices.gpu != nullptr)
      << "CoreML EP did not advertise any supported NPU or GPU device.";

  // The factory checks for duplicate NPU and GPU selections separately, so test each available device type.
  for (const OrtEpDevice* device : {coreml_devices.npu, coreml_devices.gpu}) {
    if (device == nullptr) {
      continue;
    }
    SCOPED_TRACE(testing::Message() << "device type "
                                    << Ort::GetApi().HardwareDevice_Type(Ort::GetApi().EpDevice_Device(device)));

    std::vector<Ort::ConstEpDevice> ep_devices{Ort::ConstEpDevice{device}, Ort::ConstEpDevice{device}};
    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_V2(*ort_env, ep_devices, Ort::KeyValuePairs{});

    try {
      Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);
      FAIL() << "Expected session creation to fail: the same CoreML device was selected twice.";
    } catch (const Ort::Exception& ex) {
      // The V2 registration path currently reports the factory's ORT_INVALID_ARGUMENT as ORT_FAIL, so this test
      // verifies only the diagnostic message.
      EXPECT_THAT(ex.what(), ::testing::HasSubstr("At most one device of each type can be selected"));
    }
  }
}

// Verifies that an unknown MLComputeUnits value is not rejected by the factory but reaches CoreMLOptions, which
// rejects it during provider creation, so session creation fails with the CoreMLOptions diagnostic.
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
    FAIL() << "Expected session creation to fail: MLComputeUnits=NotAComputeUnitsValue is not a valid value.";
  } catch (const Ort::Exception& ex) {
    EXPECT_THAT(ex.what(), ::testing::HasSubstr("Invalid value for option"));
    EXPECT_THAT(ex.what(), ::testing::HasSubstr("NotAComputeUnitsValue"));
  }
}

// The two tests below call CoreMLEpFactory through its EpFactoryInternal wrapper using synthetic hardware devices.
// Device advertising depends only on the device type and runtime Core ML version, so the tests require no physical
// accelerator and can run on Intel Macs that meet the version requirement. Any published OrtEpDevice instances are
// released through the EP API.
namespace {

// Creates a synthetic hardware device of the given type. The device's vendor does not affect whether the factory
// advertises the device. The Apple values below match those reported by device discovery.
OrtHardwareDevice MakeSyntheticHardwareDevice(OrtHardwareDeviceType type) {
  OrtHardwareDevice device{};
  device.type = type;
  device.vendor_id = 0x106B;  // Apple's PCI vendor ID
  device.vendor = "Apple";
  return device;
}

// Number of calls to the injected GetVersion callback.
int g_injected_version_calls = 0;

}  // namespace

// Verifies that GetSupportedDevices preserves input order, advertises at most one NPU and one GPU, ignores
// duplicate devices of either type, and excludes CPU devices.
// It publishes no more than max_ep_devices devices and leaves the remaining output entries unchanged.
// Every successful call writes the output count, including zero when no devices are advertised.
TEST(AutoEpSelection, CoreMLEPGetSupportedDevicesEnumeration) {
  if (coreml::util::CoreMLVersion() < MINIMUM_COREML_VERSION_FOR_NEURAL_ENGINE_SELECTION) {
    GTEST_SKIP() << "This test requires Core ML 6 or later because it expects the factory to advertise an NPU.";
  }

  EpFactoryInternal factory{std::make_unique<CoreMLEpFactory>()};
  OrtEpFactory* c_factory = &factory;
  ASSERT_NE(c_factory->GetSupportedDevices, nullptr);

  const OrtHardwareDevice gpu = MakeSyntheticHardwareDevice(OrtHardwareDeviceType_GPU);
  const OrtHardwareDevice npu = MakeSyntheticHardwareDevice(OrtHardwareDeviceType_NPU);
  const OrtHardwareDevice second_gpu = MakeSyntheticHardwareDevice(OrtHardwareDeviceType_GPU);
  const OrtHardwareDevice second_npu = MakeSyntheticHardwareDevice(OrtHardwareDeviceType_NPU);
  const OrtHardwareDevice cpu = MakeSyntheticHardwareDevice(OrtHardwareDeviceType_CPU);

  constexpr size_t kOutputCapacity = 5;
  OrtEpDevice untouched_marker{};

  // Calls GetSupportedDevices on the given hardware devices with max_ep_devices as the output capacity. Returns the
  // hardware device behind each published OrtEpDevice in publication order and releases the instances. Output
  // entries beyond the published count must keep their marker value.
  const auto published_hardware = [&](gsl::span<const OrtHardwareDevice* const> hardware, size_t max_ep_devices) {
    std::array<OrtEpDevice*, kOutputCapacity> ep_devices{};
    ep_devices.fill(&untouched_marker);
    size_t num_ep_devices = 42;  // a successful call must overwrite this
    Ort::Status status{c_factory->GetSupportedDevices(c_factory, hardware.data(), hardware.size(),
                                                      ep_devices.data(), max_ep_devices, &num_ep_devices)};

    InlinedVector<const OrtHardwareDevice*, 2> result;
    EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
    EXPECT_LE(num_ep_devices, max_ep_devices);
    if (!status.IsOK() || num_ep_devices > max_ep_devices) {
      return result;
    }

    for (size_t i = 0; i < ep_devices.size(); ++i) {
      if (i >= num_ep_devices) {
        EXPECT_EQ(ep_devices[i], &untouched_marker) << "Entry " << i << " lies beyond the published count.";
        continue;
      }

      EXPECT_NE(ep_devices[i], &untouched_marker) << "Published entry " << i << " was not written.";
      EXPECT_NE(ep_devices[i], nullptr) << "Published entry " << i << " is null.";
      if (ep_devices[i] == nullptr || ep_devices[i] == &untouched_marker) {
        continue;
      }

      result.push_back(Ort::GetApi().EpDevice_Device(ep_devices[i]));
      Ort::GetEpApi().ReleaseEpDevice(ep_devices[i]);
    }

    return result;
  };

  // Only the first GPU and NPU are advertised, in that order.
  const std::array<const OrtHardwareDevice*, 5> gpu_first{&gpu, &npu, &second_gpu, &second_npu, &cpu};
  EXPECT_THAT(published_hardware(gpu_first, kOutputCapacity), ::testing::ElementsAre(&gpu, &npu));

  // Skip the leading CPU device and preserve the NPU-before-GPU order.
  const std::array<const OrtHardwareDevice*, 3> cpu_first{&cpu, &npu, &gpu};
  EXPECT_THAT(published_hardware(cpu_first, kOutputCapacity), ::testing::ElementsAre(&npu, &gpu));

  // The output capacity limits the total number of published devices. A capacity of 1 keeps only the first
  // supported device. A capacity of 0 produces no devices.
  EXPECT_THAT(published_hardware(gpu_first, 1), ::testing::ElementsAre(&gpu));
  EXPECT_THAT(published_hardware(gpu_first, 0), ::testing::IsEmpty());

  // When the input contains only a CPU device, no devices are published and the output count is set to zero.
  const std::array<const OrtHardwareDevice*, 1> cpu_only{&cpu};
  EXPECT_THAT(published_hardware(cpu_only, kOutputCapacity), ::testing::IsEmpty());
}

// If the second OrtEpDevice creation fails, GetSupportedDevices must return an error, release the first device,
// and leave the output array and count unchanged, following ORT's C API convention for failed calls.
// The test replaces the GetVersion callback on EpFactoryInternal with one that returns a valid version on the
// first call and an invalid version on the second. CreateEpDevice validates each version, causing the second
// creation to fail.
TEST(AutoEpSelection, CoreMLEPGetSupportedDevicesRollsBackOnCreateFailure) {
  if (coreml::util::CoreMLVersion() < MINIMUM_COREML_VERSION_FOR_NEURAL_ENGINE_SELECTION) {
    GTEST_SKIP() << "This test requires Core ML 6 or later so the factory can advertise both an NPU and a GPU.";
  }

  EpFactoryInternal factory{std::make_unique<CoreMLEpFactory>()};
  OrtEpFactory* c_factory = &factory;
  g_injected_version_calls = 0;
  c_factory->GetVersion = [](const OrtEpFactory*) noexcept -> const char* {
    // A valid version for the first device and an invalid one for the second, so the second CreateEpDevice fails.
    return ++g_injected_version_calls == 1 ? "1.0.0" : "not a version";
  };

  const OrtHardwareDevice npu = MakeSyntheticHardwareDevice(OrtHardwareDeviceType_NPU);
  const OrtHardwareDevice gpu = MakeSyntheticHardwareDevice(OrtHardwareDeviceType_GPU);
  const std::array<const OrtHardwareDevice*, 2> hardware{&npu, &gpu};

  OrtEpDevice untouched_marker{};
  std::array<OrtEpDevice*, 2> ep_devices{&untouched_marker, &untouched_marker};
  size_t num_ep_devices = 42;
  Ort::Status status{c_factory->GetSupportedDevices(c_factory, hardware.data(), hardware.size(), ep_devices.data(),
                                                    ep_devices.size(), &num_ep_devices)};

  ASSERT_FALSE(status.IsOK())
      << "GetSupportedDevices succeeded although the second CreateEpDevice was made to fail.";
  EXPECT_EQ(g_injected_version_calls, 2)
      << "GetVersion must be called for both device creation attempts.";
  EXPECT_THAT(ep_devices, ::testing::Each(&untouched_marker));
  EXPECT_EQ(num_ep_devices, size_t{42});
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

  InlinedVector<const OrtHardwareDevice*> devices(num_devices);
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
