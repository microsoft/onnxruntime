// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_plugin_provider_interfaces.h"

#include "gsl/gsl"
#include "gtest/gtest.h"

#include "core/session/abi_devices.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime::test {

// Helper class to access public ORT APIs.
struct ApiPtrs {
  ApiPtrs() : ort_api{::OrtGetApiBase()->GetApi(ORT_API_VERSION)},
              ep_api{ort_api->GetEpApi()} {
  }

  const gsl::not_null<const ::OrtApi*> ort_api;
  const gsl::not_null<const ::OrtEpApi*> ep_api;
};

// Normally, a plugin EP would be implemented in a separate library.
// The `test_plugin_ep` namespace contains a local implementation intended for unit testing.
namespace test_plugin_ep {

struct TestOrtEp : ::OrtEp, ApiPtrs {
  TestOrtEp() : ::OrtEp{}, ApiPtrs{} {
    ort_version_supported = ORT_API_VERSION;

    GetName = GetNameImpl;

    // Individual tests should fill out the other function pointers as needed.
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEp* /*this_ptr*/) noexcept {
    constexpr const char* ep_name = "TestOrtEp";
    return ep_name;
  }
};

// This factory doesn't do anything other than implement ReleaseEp().
// It is only used to create the UniqueOrtEp that is required by PluginExecutionProvider.
struct TestOrtEpFactory : ::OrtEpFactory {
  TestOrtEpFactory() : ::OrtEpFactory{} {
    ort_version_supported = ORT_API_VERSION;
    ReleaseEp = ReleaseEpImpl;
  }

  static void ORT_API_CALL ReleaseEpImpl(::OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
    delete static_cast<TestOrtEp*>(ep);
  }
};

static TestOrtEpFactory g_test_ort_ep_factory{};

std::unique_ptr<OrtHardwareDevice> MakeTestOrtHardwareDevice(OrtHardwareDeviceType type) {
  auto hw_device = std::make_unique<OrtHardwareDevice>();
  hw_device->type = type;
  hw_device->vendor_id = 0xBE57;
  hw_device->device_id = 0;
  hw_device->vendor = "Contoso";
  return hw_device;
}

std::unique_ptr<OrtEpDevice> MakeTestOrtEpDevice(const OrtHardwareDevice* hardware_device,
                                                 const OrtMemoryInfo* device_memory_info = nullptr,
                                                 const OrtMemoryInfo* host_accessible_memory_info = nullptr) {
  auto ep_device = std::make_unique<OrtEpDevice>();
  ep_device->ep_name = "TestOrtEp";
  ep_device->ep_vendor = "Contoso";
  ep_device->device = hardware_device;
  ep_device->ep_factory = &g_test_ort_ep_factory;
  ep_device->device_memory_info = device_memory_info;
  ep_device->host_accessible_memory_info = host_accessible_memory_info;
  return ep_device;
}

OrtDevice MakeTestOrtDevice(OrtDevice::DeviceType device_type, OrtDevice::MemoryType memory_type) {
  return OrtDevice(device_type, memory_type, /*vendor_id*/ 0xBE57, /*device_id*/ 0, /*alignment*/ 16);
}

struct MakeTestOrtEpResult {
  std::unique_ptr<IExecutionProvider> ep;  // the IExecutionProvider wrapping the TestOrtEp
  gsl::not_null<TestOrtEp*> ort_ep;        // the wrapped TestOrtEp, owned by `ep`
};

// Creates an IExecutionProvider that wraps a TestOrtEp.
// The TestOrtEp is also exposed so that tests can manipulate its function pointers directly.
MakeTestOrtEpResult MakeTestOrtEp(std::vector<const OrtEpDevice*> ep_devices = {}) {
  // Default OrtHardwareDevice and OrtEpDevice used if the caller does not explicitly provide ep_devices.
  static std::unique_ptr<OrtHardwareDevice> ort_hw_device = MakeTestOrtHardwareDevice(OrtHardwareDeviceType_CPU);
  static std::unique_ptr<OrtEpDevice> ort_ep_device = MakeTestOrtEpDevice(ort_hw_device.get());

  auto ort_ep_raw = std::make_unique<TestOrtEp>().release();
  auto ort_ep = UniqueOrtEp(ort_ep_raw, OrtEpDeleter{g_test_ort_ep_factory});
  auto ort_session_options = Ort::SessionOptions{};

  if (ep_devices.empty()) {
    ep_devices.push_back(ort_ep_device.get());
  }

  auto& logging_manager = DefaultLoggingManager();
  auto ep = std::make_unique<PluginExecutionProvider>(std::move(ort_ep),
                                                      *static_cast<const OrtSessionOptions*>(ort_session_options),
                                                      g_test_ort_ep_factory,
                                                      ep_devices,
                                                      logging_manager.DefaultLogger());

  auto result = MakeTestOrtEpResult{std::move(ep), ort_ep_raw};
  return result;
}

}  // namespace test_plugin_ep

TEST(PluginExecutionProviderTest, GetPreferredLayout) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  {
    ort_ep->GetPreferredDataLayout = nullptr;
    ASSERT_EQ(ep->GetPreferredLayout(), DataLayout::NCHW);
  }

  {
    auto prefer_nhwc_fn = [](OrtEp* /*this_ptr*/, OrtEpDataLayout* preferred_data_layout) noexcept -> ::OrtStatus* {
      *preferred_data_layout = OrtEpDataLayout::OrtEpDataLayout_NCHW;
      return nullptr;
    };
    ort_ep->GetPreferredDataLayout = prefer_nhwc_fn;
    ASSERT_EQ(ep->GetPreferredLayout(), DataLayout::NCHW);
  }

#if !defined(ORT_NO_EXCEPTIONS)
  {
    auto invalid_layout_fn = [](OrtEp* /*this_ptr*/, OrtEpDataLayout* preferred_data_layout) noexcept -> ::OrtStatus* {
      *preferred_data_layout = static_cast<OrtEpDataLayout>(-1);
      return nullptr;
    };
    ort_ep->GetPreferredDataLayout = invalid_layout_fn;
    ASSERT_THROW(ep->GetPreferredLayout(), OnnxRuntimeException);
  }

  {
    auto failing_fn = [](OrtEp* this_ptr, OrtEpDataLayout* /*preferred_data_layout*/) noexcept -> ::OrtStatus* {
      auto* test_ort_ep = static_cast<test_plugin_ep::TestOrtEp*>(this_ptr);
      return test_ort_ep->ort_api->CreateStatus(OrtErrorCode::ORT_FAIL, "I can't decide what data layout I prefer.");
    };
    ort_ep->GetPreferredDataLayout = failing_fn;
    ASSERT_THROW(ep->GetPreferredLayout(), OnnxRuntimeException);
  }
#endif  // !defined(ORT_NO_EXCEPTIONS)
}

TEST(PluginExecutionProviderTest, ShouldConvertDataLayoutForOp) {
  auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp();

  {
    ort_ep->ShouldConvertDataLayoutForOp = nullptr;
    ASSERT_EQ(ep->ShouldConvertDataLayoutForOp("", "Conv", DataLayout::NHWC), std::nullopt);
  }

  {
    auto custom_nhwc_op_determination_fn = [](OrtEp* /*this_ptr*/,
                                              const char* /*node_domain*/,
                                              const char* node_op_type,
                                              OrtEpDataLayout target_data_layout,
                                              int* should_convert) noexcept -> ::OrtStatus* {
      EXPECT_EQ(target_data_layout, OrtEpDataLayout::OrtEpDataLayout_NHWC);

      if (node_op_type == std::string_view{"Conv"}) {
        *should_convert = 1;
      } else if (node_op_type == std::string_view{"BatchNormalization"}) {
        *should_convert = 0;
      } else {
        *should_convert = -1;
      }
      return nullptr;
    };
    ort_ep->ShouldConvertDataLayoutForOp = custom_nhwc_op_determination_fn;

    std::optional<bool> should_convert{};

    should_convert = ep->ShouldConvertDataLayoutForOp("", "Conv", DataLayout::NHWC);
    ASSERT_NE(should_convert, std::nullopt);
    ASSERT_EQ(*should_convert, true);

    should_convert = ep->ShouldConvertDataLayoutForOp("", "BatchNormalization", DataLayout::NHWC);
    ASSERT_NE(should_convert, std::nullopt);
    ASSERT_EQ(*should_convert, false);

    should_convert = ep->ShouldConvertDataLayoutForOp("", "GridSample", DataLayout::NHWC);
    ASSERT_EQ(should_convert, std::nullopt);
  }

#if !defined(ORT_NO_EXCEPTIONS)
  {
    auto failing_fn = [](OrtEp* this_ptr,
                         const char* /*node_domain*/,
                         const char* /*node_op_type*/,
                         OrtEpDataLayout /*target_data_layout*/,
                         int* /*should_convert*/) noexcept -> ::OrtStatus* {
      auto* test_ort_ep = static_cast<test_plugin_ep::TestOrtEp*>(this_ptr);
      return test_ort_ep->ort_api->CreateStatus(OrtErrorCode::ORT_FAIL,
                                                "To convert to NHWC or not to convert to NHWC...");
    };
    ort_ep->ShouldConvertDataLayoutForOp = failing_fn;
    ASSERT_THROW(ep->ShouldConvertDataLayoutForOp("", "Conv", DataLayout::NHWC), OnnxRuntimeException);
  }
#endif  // !defined(ORT_NO_EXCEPTIONS)
}

TEST(PluginExecutionProviderTest, InferOrtDeviceFromDeviceMemoryInfo) {
  // 1 OrtEpDevice without a device_memory_info.
  // PluginExecutionProvider should decide to use a default OrtDevice.
  {
    auto ort_hw_device = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_CPU);
    auto ort_ep_device = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device.get());
    std::vector<const OrtEpDevice*> ep_devices{ort_ep_device.get()};

    auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp(ep_devices);
    ASSERT_EQ(ep->GetOrtDeviceByMemType(OrtMemTypeDefault), OrtDevice());
  }

  // 1 OrtEpDevice with a device_memory_info.
  // PluginExecutionProvider should decide to use the OrtDevice from the device_memory_info.
  {
    auto ort_device = test_plugin_ep::MakeTestOrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT);
    auto ort_memory_info = std::make_unique<OrtMemoryInfo>("TestOrtEp GPU", OrtAllocatorType::OrtDeviceAllocator,
                                                           ort_device, OrtMemTypeDefault);

    auto ort_hw_device = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_GPU);
    auto ort_ep_device = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device.get(),
                                                             /*device_memory_info*/ ort_memory_info.get());
    std::vector<const OrtEpDevice*> ep_devices{ort_ep_device.get()};

    auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp(ep_devices);
    ASSERT_EQ(ep->GetOrtDeviceByMemType(OrtMemTypeDefault), ort_device);
  }

  // 2 OrtEpDevice instances with the same device_memory_info.
  // PluginExecutionProvider should decide to use the OrtDevice from the device_memory_info.
  {
    auto ort_device = test_plugin_ep::MakeTestOrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT);
    auto ort_memory_info = std::make_unique<OrtMemoryInfo>("TestOrtEp CPU", OrtAllocatorType::OrtDeviceAllocator,
                                                           ort_device, OrtMemTypeDefault);

    auto ort_hw_device_gpu = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_GPU);
    auto ort_hw_device_npu = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_NPU);
    auto ort_ep_device_gpu = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device_gpu.get(), ort_memory_info.get());
    auto ort_ep_device_npu = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device_npu.get(), ort_memory_info.get());
    std::vector<const OrtEpDevice*> ep_devices{ort_ep_device_gpu.get(), ort_ep_device_npu.get()};

    auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp(ep_devices);
    ASSERT_EQ(ep->GetOrtDeviceByMemType(OrtMemTypeDefault), ort_device);
  }

  // 2 OrtEpDevice instances with the different (but equivalent) device_memory_info pointers.
  // PluginExecutionProvider should decide to use a OrtDevice that is equal to the devices used by both
  // device_memory_info pointers.
  {
    auto ort_device = test_plugin_ep::MakeTestOrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT);
    auto ort_memory_info_0 = std::make_unique<OrtMemoryInfo>("TestOrtEp CPU", OrtAllocatorType::OrtDeviceAllocator,
                                                             ort_device, OrtMemTypeDefault);
    auto ort_memory_info_1 = std::make_unique<OrtMemoryInfo>("TestOrtEp CPU", OrtAllocatorType::OrtDeviceAllocator,
                                                             ort_device, OrtMemTypeDefault);

    auto ort_hw_device_gpu = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_GPU);
    auto ort_hw_device_npu = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_NPU);
    auto ort_ep_device_gpu = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device_gpu.get(), ort_memory_info_0.get());
    auto ort_ep_device_npu = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device_npu.get(), ort_memory_info_1.get());
    std::vector<const OrtEpDevice*> ep_devices{ort_ep_device_gpu.get(), ort_ep_device_npu.get()};

    auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp(ep_devices);
    ASSERT_EQ(ep->GetOrtDeviceByMemType(OrtMemTypeDefault), ort_device);
  }

  // 1 OrtEpDevice with only a host_accessible_memory_info.
  // PluginExecutionProvider should decide to use a default OrtDevice (cpu).
  {
    auto ort_device = test_plugin_ep::MakeTestOrtDevice(OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE);
    auto ort_memory_info = std::make_unique<OrtMemoryInfo>("TestOrtEp GPU", OrtAllocatorType::OrtDeviceAllocator,
                                                           ort_device, OrtMemTypeDefault);

    auto ort_hw_device = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_GPU);
    auto ort_ep_device = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device.get(),
                                                             /*device_memory_info*/ nullptr,
                                                             /*host_accessible_memory_info*/ ort_memory_info.get());
    std::vector<const OrtEpDevice*> ep_devices{ort_ep_device.get()};

    auto [ep, ort_ep] = test_plugin_ep::MakeTestOrtEp(ep_devices);
    ASSERT_EQ(ep->GetOrtDeviceByMemType(OrtMemTypeDefault), OrtDevice());
  }

#if !defined(ORT_NO_EXCEPTIONS)
  // 2 OrtEpDevice instances with DIFFERENT device_memory_info instances.
  // Should throw an exception on construction of PluginExecutionProvider.
  {
    auto ort_device_gpu = test_plugin_ep::MakeTestOrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT);
    auto ort_memory_info_gpu = std::make_unique<OrtMemoryInfo>("TestOrtEp GPU", OrtAllocatorType::OrtDeviceAllocator,
                                                               ort_device_gpu, OrtMemTypeDefault);

    auto ort_device_npu = test_plugin_ep::MakeTestOrtDevice(OrtDevice::NPU, OrtDevice::MemType::DEFAULT);
    auto ort_memory_info_npu = std::make_unique<OrtMemoryInfo>("TestOrtEp NPU", OrtAllocatorType::OrtDeviceAllocator,
                                                               ort_device_npu, OrtMemTypeDefault);

    auto ort_hw_device_gpu = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_GPU);
    auto ort_hw_device_npu = test_plugin_ep::MakeTestOrtHardwareDevice(OrtHardwareDeviceType_NPU);
    auto ort_ep_device_gpu = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device_gpu.get(), ort_memory_info_gpu.get());
    auto ort_ep_device_npu = test_plugin_ep::MakeTestOrtEpDevice(ort_hw_device_npu.get(), ort_memory_info_npu.get());
    std::vector<const OrtEpDevice*> ep_devices{ort_ep_device_gpu.get(), ort_ep_device_npu.get()};

    ASSERT_THROW(test_plugin_ep::MakeTestOrtEp(ep_devices), OnnxRuntimeException);
  }
#endif  // !defined(ORT_NO_EXCEPTIONS)
}

}  // namespace onnxruntime::test
