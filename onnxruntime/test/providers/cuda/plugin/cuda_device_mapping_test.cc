// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <array>
#include <cstdint>
#include <string>

#include "gtest/gtest.h"

#include "core/providers/cuda/plugin/cuda_device_mapping.h"

namespace onnxruntime::cuda_plugin::test {
namespace {

TEST(CudaDeviceMappingTest, MatchesReorderedDevicesByPciBusId) {
  const std::array<std::string, 2> cuda_pci_bus_ids{"0000:02:00.0", "0000:01:00.0"};
  std::array<uint8_t, 2> assigned{};

  EXPECT_EQ(FindCudaOrdinalForHardwareDeviceIdentity("0000:01:00.0", cuda_pci_bus_ids, assigned), 1);
}

TEST(CudaDeviceMappingTest, LeavesMissingPrefixOrdinalForRuntimeDiscovery) {
  const std::array<std::string, 2> cuda_pci_bus_ids{"0000:01:00.0", "0000:02:00.0"};
  std::array<uint8_t, 2> assigned{};

  auto ordinal = FindCudaOrdinalForHardwareDeviceIdentity("0000:02:00.0", cuda_pci_bus_ids, assigned);
  ASSERT_EQ(ordinal, 1);
  assigned[*ordinal] = 1;

  EXPECT_EQ(assigned[0], 0);
}

TEST(CudaDeviceMappingTest, DoesNotAssignKnownHiddenHardwareDevice) {
  const std::array<std::string, 1> cuda_pci_bus_ids{"0000:01:00.0"};
  const std::array<uint8_t, 1> assigned{};

  EXPECT_EQ(FindCudaOrdinalForHardwareDeviceIdentity("0000:02:00.0", cuda_pci_bus_ids, assigned),
            std::nullopt);
}

TEST(CudaDeviceMappingTest, MatchesDuplicateMigPciBusIdsToDistinctOrdinals) {
  const std::array<std::string, 2> cuda_pci_bus_ids{"0000:01:00.0", "0000:01:00.0"};
  std::array<uint8_t, 2> assigned{};

  auto first_ordinal = FindCudaOrdinalForHardwareDeviceIdentity("0000:01:00.0", cuda_pci_bus_ids, assigned);
  ASSERT_EQ(first_ordinal, 0);
  assigned[*first_ordinal] = 1;

  EXPECT_EQ(FindCudaOrdinalForHardwareDeviceIdentity("0000:01:00.0", cuda_pci_bus_ids, assigned), 1);
}

TEST(CudaDeviceMappingTest, PositionalFallbackOnlyUsesCudaDevicesWithoutIdentity) {
  const std::array<std::string, 2> cuda_pci_bus_ids{"", ""};
  std::array<uint8_t, 2> assigned{};

  auto first_ordinal = FindCudaOrdinalWithoutIdentity(cuda_pci_bus_ids, assigned);
  ASSERT_EQ(first_ordinal, 0);
  assigned[*first_ordinal] = 1;

  EXPECT_EQ(FindCudaOrdinalWithoutIdentity(cuda_pci_bus_ids, assigned), 1);
}

TEST(CudaDeviceMappingTest, UnknownHardwareCannotStealKnownCudaIdentity) {
  const std::array<std::string, 2> cuda_device_identities{"0000:01:00.0", ""};
  const std::array<uint8_t, 2> assigned{};

  EXPECT_EQ(FindCudaOrdinalWithoutIdentity(cuda_device_identities, assigned), 1);
}

TEST(CudaDeviceMappingTest, ExactMatchIsReservedBeforeUnknownHardwareFallback) {
  const std::array<std::string, 2> cuda_device_identities{"0000:02:00.0", ""};
  std::array<uint8_t, 2> assigned{};

  auto exact_ordinal =
      FindCudaOrdinalForHardwareDeviceIdentity("0000:02:00.0", cuda_device_identities, assigned);
  ASSERT_EQ(exact_ordinal, 0);
  assigned[*exact_ordinal] = 1;

  EXPECT_EQ(FindCudaOrdinalWithoutIdentity(cuda_device_identities, assigned), 1);
}

}  // namespace
}  // namespace onnxruntime::cuda_plugin::test

#endif  // defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
