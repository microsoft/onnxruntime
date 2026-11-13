// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/environment.h"
#include "core/session/plugin_ep/ep_factory_internal.h"
#include "core/session/plugin_ep/ep_factory_provider_bridge.h"
#include "core/session/provider_bridge_library.h"
#include "core/session/provider_policy_context.h"
#include "test/framework/dummy_provider.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

#if defined(USE_DML)
#include "core/session/plugin_ep/ep_factory_dml.h"
#endif

namespace onnxruntime {
namespace test {

namespace {
// Records the hardware devices and EP metadata passed by EpFactoryInternal.
// When create_provider is false, returns success without creating a provider.
class RecordingFactoryImpl : public EpFactoryInternalImpl {
 public:
  explicit RecordingFactoryImpl(bool create_provider = true)
      : EpFactoryInternalImpl("RecordingTestEp", "Contoso", 0x1234), create_provider_{create_provider} {}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(RecordingFactoryImpl);

  OrtStatus* GetSupportedDevices(EpFactoryInternal& /*ep_factory*/,
                                 const OrtHardwareDevice* const* /*devices*/,
                                 size_t /*num_devices*/,
                                 OrtEpDevice** /*ep_devices*/,
                                 size_t /*max_ep_devices*/,
                                 size_t* num_ep_devices) noexcept override {
    *num_ep_devices = 0;
    return nullptr;
  }

  OrtStatus* CreateIExecutionProvider(const OrtHardwareDevice* const* devices,
                                      const OrtKeyValuePairs* const* ep_metadata_pairs,
                                      size_t num_devices,
                                      const OrtSessionOptions* /*session_options*/,
                                      const OrtLogger* /*logger*/,
                                      std::unique_ptr<IExecutionProvider>* ep) override {
    received_hardware_devices.assign(devices, devices + num_devices);
    received_ep_metadata.assign(ep_metadata_pairs, ep_metadata_pairs + num_devices);

    *ep = create_provider_ ? std::make_unique<DummyExecutionProvider>() : nullptr;
    return nullptr;
  }

  InlinedVector<const OrtHardwareDevice*> received_hardware_devices;
  InlinedVector<const OrtKeyValuePairs*> received_ep_metadata;

 private:
  const bool create_provider_;
};
}  // namespace

// Verifies that InternalExecutionProviderFactory, used by SessionOptionsAppendExecutionProvider_V2, preserves
// the selection. The provider must report all selected OrtEpDevice instances in order, and the implementation
// must receive the corresponding hardware devices and EP metadata in the same order.
TEST(InternalEpDevices, FactoryPassesTheWholeSelectionToTheProvider) {
  OrtHardwareDevice first_hardware{};
  OrtHardwareDevice second_hardware{};
  OrtEpDevice first{};
  OrtEpDevice second{};
  first.device = &first_hardware;
  second.device = &second_hardware;
  std::array<const OrtEpDevice*, 2> selection{&first, &second};

  auto impl = std::make_unique<RecordingFactoryImpl>();
  RecordingFactoryImpl& impl_ref = *impl;
  EpFactoryInternal factory{std::move(impl)};

  OrtSessionOptions session_options;
  const OrtLogger& logger = *logging::LoggingManager::DefaultLogger().ToExternal();
  InternalExecutionProviderFactory provider_factory{factory, selection};
  // The caller's array may change or be destroyed after the factory is constructed, so the factory must copy it.
  selection = {&second, &first};
  const std::unique_ptr<IExecutionProvider> ep = provider_factory.CreateProvider(session_options, logger);
  ASSERT_NE(ep, nullptr);

  EXPECT_THAT(ep->GetEpDevices(), ::testing::ElementsAre(&first, &second));
  EXPECT_THAT(impl_ref.received_hardware_devices, ::testing::ElementsAre(&first_hardware, &second_hardware));
  EXPECT_THAT(impl_ref.received_ep_metadata, ::testing::ElementsAre(&first.ep_metadata, &second.ep_metadata));
}

TEST(InternalEpDevices, FactoryHandlesImplThatCreatesNoProvider) {
  OrtEpDevice device{};
  const std::array<const OrtEpDevice*, 1> selection{&device};
  EpFactoryInternal factory{std::make_unique<RecordingFactoryImpl>(/*create_provider=*/false)};

  OrtSessionOptions session_options;
  const OrtLogger& logger = *logging::LoggingManager::DefaultLogger().ToExternal();
  std::unique_ptr<IExecutionProvider> ep;
  ASSERT_STATUS_OK(ToStatusAndRelease(factory.CreateIExecutionProvider(selection, &session_options, &logger, &ep)));
  EXPECT_EQ(ep, nullptr);
}

#if !defined(ORT_NO_EXCEPTIONS)
// ProviderLibrary::Get throws if the provider library cannot be loaded. Verify that the factory's noexcept
// CreateIExecutionProvider returns an error status instead of terminating the process.
TEST(InternalEpDevices, ProviderBridgeFactoryReturnsStatusWhenCreationThrows) {
  OrtEpFactory library_factory{};
  library_factory.GetName = [](const OrtEpFactory* /*this_ptr*/) noexcept -> const char* { return "TestBridgeEp"; };
  library_factory.GetVendor = [](const OrtEpFactory* /*this_ptr*/) noexcept -> const char* { return "Contoso"; };
  library_factory.GetVendorId = [](const OrtEpFactory* /*this_ptr*/) noexcept -> uint32_t { return 0x1234; };
  ProviderLibrary provider_library{ORT_TSTR("onnxruntime_providers_missing_for_test")};
  EpFactoryInternal factory{std::make_unique<ProviderBridgeEpFactory>(library_factory, provider_library)};

  OrtEpDevice device{};
  const std::array<const OrtEpDevice*, 1> selection{&device};
  OrtSessionOptions session_options;
  const OrtLogger& logger = *logging::LoggingManager::DefaultLogger().ToExternal();
  std::unique_ptr<IExecutionProvider> ep;
  const Status status = ToStatusAndRelease(factory.CreateIExecutionProvider(selection, &session_options, &logger, &ep));
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(ep, nullptr);
}

#if defined(USE_DML)
// An invalid device_id causes option parsing to throw before the DML factory accesses a GPU,
// so this test does not require a GPU.
TEST(InternalEpDevices, DmlFactoryReturnsStatusForInvalidOption) {
  EpFactoryInternal factory{std::make_unique<DmlEpFactory>()};

  OrtEpDevice device{};
  const std::array<const OrtEpDevice*, 1> selection{&device};
  OrtSessionOptions session_options;
  const std::string device_id_key = OrtSessionOptions::GetProviderOptionPrefix(kDmlExecutionProvider) + "device_id";
  ASSERT_STATUS_OK(session_options.value.config_options.AddConfigEntry(device_id_key.c_str(), "invalid"));
  const OrtLogger& logger = *logging::LoggingManager::DefaultLogger().ToExternal();
  std::unique_ptr<IExecutionProvider> ep;
  const Status status = ToStatusAndRelease(factory.CreateIExecutionProvider(selection, &session_options, &logger, &ep));
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(ep, nullptr);
}
#endif  // defined(USE_DML)
#endif  // !defined(ORT_NO_EXCEPTIONS)

// The ORT CPU EP has an internal factory in every non-minimal build, so this test does not require an accelerator.
TEST(InternalEpDevices, PolicyCreatedProviderCarriesItsSelection) {
  const Environment& env = GetEnvironment();
  const OrtEpDevice* cpu_ep_device = nullptr;
  for (const OrtEpDevice* ep_device : env.GetOrtEpDevices()) {
    if (ep_device->ep_name == kCpuExecutionProvider) {
      cpu_ep_device = ep_device;
      break;
    }
  }
  ASSERT_NE(cpu_ep_device, nullptr) << "The ORT CPU EP did not publish an OrtEpDevice.";

  ProviderPolicyContext context;
  std::vector<SelectionInfo> selections;
  context.FoldSelectedDevices({cpu_ep_device}, selections);
  ASSERT_EQ(selections.size(), 1U);

  OrtSessionOptions session_options;
  const OrtLogger& logger = *logging::LoggingManager::DefaultLogger().ToExternal();
  std::unique_ptr<IExecutionProvider> ep;
  ASSERT_STATUS_OK(context.CreateExecutionProvider(env, session_options, logger, selections[0], ep));
  ASSERT_NE(ep, nullptr);
  EXPECT_THAT(ep->GetEpDevices(), ::testing::ElementsAre(cpu_ep_device));
}

}  // namespace test
}  // namespace onnxruntime
