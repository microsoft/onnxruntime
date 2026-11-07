// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/plugin_ep/ep_factory_internal.h"
#include "gtest/gtest.h"

namespace onnxruntime::test {
namespace {

struct InteropFactory : OrtEpFactory {
  InteropFactory() : OrtEpFactory{} {
    ort_version_supported = ORT_API_VERSION;
    SynchronizeDevice = [](OrtEpFactory* self, int device_id) noexcept -> OrtStatus* {
      static_cast<InteropFactory*>(self)->last_sync_device = device_id;
      return nullptr;
    };
    GetCurrentDevice = [](OrtEpFactory* self, int* device_id) noexcept -> OrtStatus* {
      *device_id = static_cast<InteropFactory*>(self)->current_device;
      return nullptr;
    };
    SetCurrentDevice = [](OrtEpFactory* self, int device_id) noexcept -> OrtStatus* {
      static_cast<InteropFactory*>(self)->current_device = device_id;
      return nullptr;
    };
  }
  int current_device = 7;
  int last_sync_device = -1;
};

class InteropFactoryWrapper final : public EpFactoryInternalImpl {
 public:
  explicit InteropFactoryWrapper(OrtEpFactory& factory)
      : EpFactoryInternalImpl("InteropTest", "Test", 0), factory_(factory) {}

  OrtEpFactory* GetDeviceInteropFactory() const noexcept override { return &factory_; }

  OrtStatus* GetSupportedDevices(EpFactoryInternal&, const OrtHardwareDevice* const*, size_t,
                                 OrtEpDevice**, size_t, size_t* count) noexcept override {
    *count = 0;
    return nullptr;
  }

  OrtStatus* CreateIExecutionProvider(const OrtHardwareDevice* const*, const OrtKeyValuePairs* const*, size_t,
                                      const OrtSessionOptions*, const OrtLogger*,
                                      std::unique_ptr<IExecutionProvider>*) override {
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Interop test factory does not create execution providers.");
  }

 private:
  OrtEpFactory& factory_;
};

TEST(EpDeviceInteropTest, InternalFactoryForwardsSupportedCallbacks) {
  InteropFactory inner;
  EpFactoryInternal wrapper(std::make_unique<InteropFactoryWrapper>(inner));
  auto& factory = static_cast<OrtEpFactory&>(wrapper);
  ASSERT_NE(factory.SynchronizeDevice, nullptr);
  ASSERT_NE(factory.GetCurrentDevice, nullptr);
  ASSERT_NE(factory.SetCurrentDevice, nullptr);
  int current_device = -1;
  EXPECT_EQ(factory.GetCurrentDevice(&factory, &current_device), nullptr);
  EXPECT_EQ(current_device, 7);
  EXPECT_EQ(factory.SetCurrentDevice(&factory, 2), nullptr);
  EXPECT_EQ(inner.current_device, 2);
  EXPECT_EQ(factory.SynchronizeDevice(&factory, 2), nullptr);
  EXPECT_EQ(inner.last_sync_device, 2);
}

TEST(EpDeviceInteropTest, InternalFactoryPreservesOptionalAndVersionGuards) {
  for (const bool old_version : {false, true}) {
    InteropFactory inner;
    if (old_version) {
      inner.ort_version_supported = 31;
    } else {
      inner.SetCurrentDevice = nullptr;
      inner.SynchronizeDevice = nullptr;
    }
    EpFactoryInternal wrapper(std::make_unique<InteropFactoryWrapper>(inner));
    auto& factory = static_cast<OrtEpFactory&>(wrapper);
    EXPECT_EQ(factory.SynchronizeDevice, nullptr);
    EXPECT_EQ(factory.GetCurrentDevice, nullptr);
    EXPECT_EQ(factory.SetCurrentDevice, nullptr);
  }
}

}  // namespace
}  // namespace onnxruntime::test

#endif
