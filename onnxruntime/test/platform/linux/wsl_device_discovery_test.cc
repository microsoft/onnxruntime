// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/linux/wsl_device_discovery.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/pci_vendor_ids.h"
#include "gtest/gtest.h"
#include "test/util/include/asserts.h"

namespace fs = std::filesystem;

namespace onnxruntime::test {

namespace {

struct FakeDxCore {
  std::vector<wsl_device_discovery::D3DKMT_ADAPTERINFO> adapters;
  std::unordered_map<uint32_t, wsl_device_discovery::D3DKMT_DEVICE_IDS> device_ids;
  std::unordered_set<uint32_t> fail_query_handles;

  int32_t enum3_status = 0;
  int32_t enum2_status = 0;
  bool enum3_report_empty = false;
  bool enable_enum3 = true;
  bool enable_enum2 = true;
  bool enable_query = true;
  bool enable_close = true;

  int enum3_calls = 0;
  int enum2_calls = 0;
  bool enum3_counted = false;
  bool enum3_filled = false;
  uint64_t enum3_filter = 0;
  std::vector<uint32_t> queried_handles;
  std::vector<uint32_t> query_types;
  std::vector<uint32_t> query_sizes;
  std::vector<uint32_t> closed_handles;
};

FakeDxCore* g_fake = nullptr;

int32_t FillAdapters(uint32_t& num_adapters,
                       wsl_device_discovery::D3DKMT_ADAPTERINFO* dest,
                       bool& counted,
                       bool& filled) {
  if (dest == nullptr) {
    counted = true;
    num_adapters = static_cast<uint32_t>(g_fake->adapters.size());
    return 0;
  }

  filled = true;
  const uint32_t count = std::min(num_adapters, static_cast<uint32_t>(g_fake->adapters.size()));
  for (uint32_t i = 0; i < count; ++i) {
    dest[i] = g_fake->adapters[i];
  }
  num_adapters = count;
  return 0;
}

int32_t FakeEnumAdapters3(void* args) {
  auto* enum_args = static_cast<wsl_device_discovery::D3DKMT_ENUMADAPTERS3*>(args);
  ++g_fake->enum3_calls;
  g_fake->enum3_filter = enum_args->Filter;
  if (g_fake->enum3_status != 0) {
    return g_fake->enum3_status;
  }
  if (g_fake->enum3_report_empty) {
    g_fake->enum3_counted = enum_args->pAdapters == nullptr;
    g_fake->enum3_filled = enum_args->pAdapters != nullptr;
    enum_args->NumAdapters = 0;
    return 0;
  }
  return FillAdapters(enum_args->NumAdapters, enum_args->pAdapters, g_fake->enum3_counted, g_fake->enum3_filled);
}

int32_t FakeEnumAdapters2(void* args) {
  auto* enum_args = static_cast<wsl_device_discovery::D3DKMT_ENUMADAPTERS2*>(args);
  ++g_fake->enum2_calls;
  if (g_fake->enum2_status != 0) {
    return g_fake->enum2_status;
  }
  bool counted = false;
  bool filled = false;
  return FillAdapters(enum_args->NumAdapters, enum_args->pAdapters, counted, filled);
}

int32_t FakeQueryAdapterInfo(void* args) {
  auto* query = static_cast<wsl_device_discovery::D3DKMT_QUERYADAPTERINFO*>(args);
  g_fake->queried_handles.push_back(query->hAdapter);
  g_fake->query_types.push_back(query->Type);
  g_fake->query_sizes.push_back(query->PrivateDriverDataSize);
  if (g_fake->fail_query_handles.count(query->hAdapter) != 0) {
    return 1;
  }

  auto* device_ids = static_cast<wsl_device_discovery::D3DKMT_QUERY_DEVICE_IDS*>(query->pPrivateDriverData);
  device_ids->DeviceIds = g_fake->device_ids.at(query->hAdapter);
  return 0;
}

int32_t FakeCloseAdapter(void* args) {
  auto* close_args = static_cast<wsl_device_discovery::D3DKMT_CLOSEADAPTER*>(args);
  g_fake->closed_handles.push_back(close_args->hAdapter);
  return 0;
}

wsl_device_discovery::DxCoreThunks MakeThunks(FakeDxCore& fake) {
  g_fake = &fake;
  wsl_device_discovery::DxCoreThunks thunks{};
  if (fake.enable_enum3) {
    thunks.enum_adapters3 = FakeEnumAdapters3;
  }
  if (fake.enable_enum2) {
    thunks.enum_adapters2 = FakeEnumAdapters2;
  }
  if (fake.enable_query) {
    thunks.query_adapter_info = FakeQueryAdapterInfo;
  }
  if (fake.enable_close) {
    thunks.close_adapter = FakeCloseAdapter;
  }
  return thunks;
}

wsl_device_discovery::D3DKMT_ADAPTERINFO MakeAdapter(uint32_t handle, uint32_t luid_low, int32_t luid_high) {
  wsl_device_discovery::D3DKMT_ADAPTERINFO adapter{};
  adapter.hAdapter = handle;
  adapter.AdapterLuid.LowPart = luid_low;
  adapter.AdapterLuid.HighPart = luid_high;
  return adapter;
}

wsl_device_discovery::D3DKMT_DEVICE_IDS MakeDeviceIds(uint32_t vendor_id, uint32_t device_id) {
  wsl_device_discovery::D3DKMT_DEVICE_IDS ids{};
  ids.VendorID = vendor_id;
  ids.DeviceID = device_id;
  return ids;
}

OrtHardwareDevice MakeGpu(uint32_t vendor_id, uint32_t device_id, const char* card_idx) {
  OrtHardwareDevice gpu{};
  gpu.type = OrtHardwareDeviceType_GPU;
  gpu.vendor_id = vendor_id;
  gpu.device_id = device_id;
  if (card_idx != nullptr) {
    gpu.metadata.Add("card_idx", card_idx);
  }
  return gpu;
}

}  // namespace

TEST(WslDeviceDiscoveryTest, PrefersEnumAdapters3AndConvertsLuid) {
  constexpr uint32_t kHandle = 7;
  constexpr uint32_t kLuidLow = 0x89abcdef;
  constexpr int32_t kLuidHigh = -1;
  constexpr uint64_t kLuid = 0xffffffff89abcdefULL;

  FakeDxCore fake{};
  fake.adapters.push_back(MakeAdapter(kHandle, kLuidLow, kLuidHigh));
  fake.device_ids.emplace(kHandle, MakeDeviceIds(pci_vendor_ids::kAmdAti, 0x744c));

  std::vector<wsl_device_discovery::WslGpuInfo> devices;
  ASSERT_STATUS_OK(wsl_device_discovery::GetGpuDevices(MakeThunks(fake), devices));

  EXPECT_EQ(fake.enum3_calls, 2);
  EXPECT_TRUE(fake.enum3_counted);
  EXPECT_TRUE(fake.enum3_filled);
  EXPECT_EQ(fake.enum3_filter, wsl_device_discovery::kEnumAdaptersFilterIncludeComputeOnly);
  EXPECT_EQ(fake.enum2_calls, 0);
  ASSERT_EQ(fake.query_types.size(), 1u);
  EXPECT_EQ(fake.query_types[0], wsl_device_discovery::kQueryTypePhysicalAdapterDeviceIds);
  EXPECT_EQ(fake.query_sizes[0], sizeof(wsl_device_discovery::D3DKMT_QUERY_DEVICE_IDS));
  EXPECT_EQ(fake.closed_handles, std::vector<uint32_t>{kHandle});

  ASSERT_EQ(devices.size(), 1u);
  EXPECT_EQ(devices[0].vendor_id, pci_vendor_ids::kAmdAti);
  EXPECT_EQ(devices[0].device_id, 0x744c);
  EXPECT_EQ(devices[0].luid, kLuid);
}

TEST(WslDeviceDiscoveryTest, FallsBackToEnumAdapters2WhenEnumAdapters3Fails) {
  constexpr uint32_t kHandle = 3;

  FakeDxCore fake{};
  fake.enum3_status = 1;
  fake.adapters.push_back(MakeAdapter(kHandle, 0x11, 0x22));
  fake.device_ids.emplace(kHandle, MakeDeviceIds(pci_vendor_ids::kNvidia, 0x2684));

  std::vector<wsl_device_discovery::WslGpuInfo> devices;
  ASSERT_STATUS_OK(wsl_device_discovery::GetGpuDevices(MakeThunks(fake), devices));

  EXPECT_EQ(fake.enum3_calls, 1);
  EXPECT_EQ(fake.enum2_calls, 2);
  ASSERT_EQ(devices.size(), 1u);
  EXPECT_EQ(devices[0].vendor_id, pci_vendor_ids::kNvidia);
  EXPECT_EQ(devices[0].device_id, 0x2684);
  EXPECT_EQ(devices[0].luid, (static_cast<uint64_t>(0x22u) << 32) | 0x11u);
  EXPECT_EQ(fake.closed_handles, std::vector<uint32_t>{kHandle});
}

TEST(WslDeviceDiscoveryTest, DoesNotFallBackWhenEnumAdapters3ReportsNoAdapters) {
  constexpr uint32_t kHandle = 1;

  FakeDxCore fake{};
  // EnumAdapters2 would still see this adapter. An empty EnumAdapters3 result must not use it.
  fake.adapters.push_back(MakeAdapter(kHandle, 1, 1));
  fake.device_ids.emplace(kHandle, MakeDeviceIds(pci_vendor_ids::kAmdAti, 0x1));
  fake.enum3_report_empty = true;

  std::vector<wsl_device_discovery::WslGpuInfo> devices;
  ASSERT_STATUS_OK(wsl_device_discovery::GetGpuDevices(MakeThunks(fake), devices));

  EXPECT_EQ(fake.enum3_calls, 1);
  EXPECT_TRUE(fake.enum3_counted);
  EXPECT_FALSE(fake.enum3_filled);
  EXPECT_EQ(fake.enum2_calls, 0);
  EXPECT_TRUE(devices.empty());
}

TEST(WslDeviceDiscoveryTest, ClosesHandleWhenAdapterQueryFails) {
  constexpr uint32_t kFailedHandle = 10;
  constexpr uint32_t kOkHandle = 11;

  FakeDxCore fake{};
  fake.adapters.push_back(MakeAdapter(kFailedHandle, 1, 2));
  fake.adapters.push_back(MakeAdapter(kOkHandle, 3, 4));
  fake.device_ids.emplace(kFailedHandle, MakeDeviceIds(pci_vendor_ids::kAmdAti, 0x1));
  fake.device_ids.emplace(kOkHandle, MakeDeviceIds(pci_vendor_ids::kAmdAti, 0x2));
  fake.fail_query_handles.insert(kFailedHandle);

  std::vector<wsl_device_discovery::WslGpuInfo> devices;
  ASSERT_STATUS_OK(wsl_device_discovery::GetGpuDevices(MakeThunks(fake), devices));

  EXPECT_EQ(fake.queried_handles, (std::vector<uint32_t>{kFailedHandle, kOkHandle}));
  EXPECT_EQ(fake.closed_handles, (std::vector<uint32_t>{kFailedHandle, kOkHandle}));
  ASSERT_EQ(devices.size(), 1u);
  EXPECT_EQ(devices[0].device_id, 0x2);
}

TEST(WslDeviceDiscoveryTest, SkipsEnumerationWhenCloseAdapterIsMissing) {
  FakeDxCore fake{};
  fake.enable_close = false;
  fake.adapters.push_back(MakeAdapter(1, 1, 1));
  fake.device_ids.emplace(1, MakeDeviceIds(pci_vendor_ids::kAmdAti, 0x1));

  const auto thunks = MakeThunks(fake);
  EXPECT_FALSE(thunks.IsAvailable());

  std::vector<wsl_device_discovery::WslGpuInfo> devices;
  ASSERT_STATUS_OK(wsl_device_discovery::GetGpuDevices(thunks, devices));

  EXPECT_TRUE(devices.empty());
  EXPECT_EQ(fake.enum3_calls, 0);
  EXPECT_EQ(fake.enum2_calls, 0);
  EXPECT_TRUE(fake.closed_handles.empty());
}

TEST(WslDeviceDiscoveryTest, MissingDxgDeviceYieldsNoGpus) {
  const fs::path missing = fs::temp_directory_path() / "ort_wsl_discovery_missing_dxg";
  fs::remove_all(missing);

  std::vector<wsl_device_discovery::WslGpuInfo> devices;
  ASSERT_STATUS_OK(wsl_device_discovery::GetGpuDevices(devices, missing));
  EXPECT_TRUE(devices.empty());
}

TEST(WslDeviceDiscoveryTest, KeepsSysfsDevicesWhenWslDiscoveryIsEmpty) {
  std::vector<OrtHardwareDevice> devices;
  devices.push_back(MakeGpu(pci_vendor_ids::kAmdAti, 0x744c, "0"));
  devices.push_back(MakeGpu(pci_vendor_ids::kMicrosoft, 0x008e, "1"));

  wsl_device_discovery::ApplyWslGpuDevices(devices, {});

  ASSERT_EQ(devices.size(), 2u);
  EXPECT_EQ(devices[0].vendor_id, pci_vendor_ids::kAmdAti);
  EXPECT_EQ(devices[0].metadata.Entries().at("card_idx"), "0");
  EXPECT_EQ(devices[1].vendor_id, pci_vendor_ids::kMicrosoft);
  EXPECT_EQ(devices[1].device_id, 0x008eu);
  EXPECT_EQ(devices[1].metadata.Entries().at("card_idx"), "1");
}

TEST(WslDeviceDiscoveryTest, ReplacesMicrosoftPlaceholderAndKeepsOtherSysfsDevices) {
  constexpr uint64_t kLuid = 0x100000002ULL;

  std::vector<OrtHardwareDevice> devices;
  devices.push_back(MakeGpu(pci_vendor_ids::kAmdAti, 0x13c0, "2"));
  devices.push_back(MakeGpu(pci_vendor_ids::kMicrosoft, 0x008e, "3"));

  wsl_device_discovery::WslGpuInfo wsl_gpu{};
  wsl_gpu.vendor_id = static_cast<uint16_t>(pci_vendor_ids::kNvidia);
  wsl_gpu.device_id = 0x2684;
  wsl_gpu.luid = kLuid;

  wsl_device_discovery::ApplyWslGpuDevices(devices, {wsl_gpu});

  ASSERT_EQ(devices.size(), 2u);
  EXPECT_EQ(devices[0].vendor_id, pci_vendor_ids::kAmdAti);
  EXPECT_EQ(devices[0].metadata.Entries().at("card_idx"), "2");
  EXPECT_EQ(devices[1].vendor_id, pci_vendor_ids::kNvidia);
  EXPECT_EQ(devices[1].device_id, 0x2684u);
  EXPECT_EQ(devices[1].metadata.Entries().at("LUID"), std::to_string(kLuid));
  EXPECT_EQ(devices[1].metadata.Entries().at("Discrete"), "1");
  EXPECT_EQ(devices[1].metadata.Entries().find("card_idx"), devices[1].metadata.Entries().end());
}

}  // namespace onnxruntime::test
