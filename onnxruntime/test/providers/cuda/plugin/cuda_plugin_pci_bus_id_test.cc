// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests that the CUDA plugin EP correctly resolves CUDA ordinals via pci_bus_id
// metadata from OrtHardwareDevice, using cudaDeviceGetByPCIBusId.

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/file_util.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

constexpr const char* kCudaPluginEpRegistrationName = "CudaPluginPciBusIdTest";

// Resolve the CUDA plugin EP shared library path.
std::filesystem::path GetCudaPluginLibraryPath() {
  return GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_cuda_plugin"));
}

// RAII handle that registers/unregisters the CUDA plugin EP library.
class ScopedCudaPluginRegistration {
 public:
  ScopedCudaPluginRegistration(Ort::Env& env, const char* registration_name)
      : env_(env), name_(registration_name) {
    auto lib_path = GetCudaPluginLibraryPath();
    if (!std::filesystem::exists(lib_path)) {
      available_ = false;
      return;
    }
    env_.RegisterExecutionProviderLibrary(name_.c_str(), lib_path.c_str());
    available_ = true;
  }

  ~ScopedCudaPluginRegistration() {
    if (available_) {
      try {
        env_.UnregisterExecutionProviderLibrary(name_.c_str());
      } catch (...) {
      }
    }
  }

  bool IsAvailable() const { return available_; }

  ScopedCudaPluginRegistration(const ScopedCudaPluginRegistration&) = delete;
  ScopedCudaPluginRegistration& operator=(const ScopedCudaPluginRegistration&) = delete;

 private:
  Ort::Env& env_;
  std::string name_;
  bool available_ = false;
};

}  // namespace

class CudaPluginPciBusIdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaError_t err = cudaGetDeviceCount(&cuda_device_count_);
    if (err != cudaSuccess || cuda_device_count_ == 0) {
      GTEST_SKIP() << "No CUDA device available.";
    }

    registration_ = std::make_unique<ScopedCudaPluginRegistration>(
        *ort_env, kCudaPluginEpRegistrationName);
    if (!registration_->IsAvailable()) {
      GTEST_SKIP() << "CUDA plugin EP library not found.";
    }
  }

  void TearDown() override {
    registration_.reset();
    cudaDeviceSynchronize();
  }

  std::unique_ptr<ScopedCudaPluginRegistration> registration_;
  int cuda_device_count_ = 0;
};

// Verify that the CUDA plugin EP device's cuda_device_id in ep_metadata matches
// the ordinal obtained by calling cudaDeviceGetByPCIBusId with the pci_bus_id
// from the hardware device metadata.
TEST_F(CudaPluginPciBusIdTest, CudaOrdinalMatchesPciBusIdResolution) {
  auto ep_devices = ort_env->GetEpDevices();

  int cuda_ep_device_count = 0;
  for (const auto& ep_device : ep_devices) {
    if (strcmp(ep_device.EpName(), "CudaPluginExecutionProvider") != 0) {
      continue;
    }

    ++cuda_ep_device_count;

    // Get the pci_bus_id from the hardware device metadata.
    auto hw_device = ep_device.Device();
    ASSERT_TRUE(hw_device) << "EpDevice has no associated hardware device.";

    auto hw_metadata = hw_device.Metadata();
    const char* pci_bus_id = hw_metadata.GetValue("pci_bus_id");

    // pci_bus_id may not be available on all platforms (e.g., some VMs).
    // If it's missing, we can't verify the mapping but the EP should still work
    // via the fallback counter-based ordinal assignment.
    if (pci_bus_id == nullptr || pci_bus_id[0] == '\0') {
      GTEST_LOG_(INFO) << "pci_bus_id not available in hardware device metadata; "
                       << "skipping PCI bus ID resolution check for this device.";
      continue;
    }

    // Resolve the CUDA ordinal from pci_bus_id using the CUDA runtime API.
    int expected_ordinal = -1;
    cudaError_t cuda_err = cudaDeviceGetByPCIBusId(&expected_ordinal, pci_bus_id);
    ASSERT_EQ(cuda_err, cudaSuccess)
        << "cudaDeviceGetByPCIBusId failed for pci_bus_id=\"" << pci_bus_id
        << "\": " << cudaGetErrorString(cuda_err);
    ASSERT_GE(expected_ordinal, 0);
    ASSERT_LT(expected_ordinal, cuda_device_count_);

    // Get the cuda_device_id that the plugin EP assigned (stored in ep_metadata).
    auto ep_metadata = ep_device.EpMetadata();
    const char* cuda_device_id_str = ep_metadata.GetValue("cuda_device_id");
    ASSERT_NE(cuda_device_id_str, nullptr)
        << "cuda_device_id missing from CUDA plugin EP metadata.";

    int actual_ordinal = std::stoi(cuda_device_id_str);

    EXPECT_EQ(actual_ordinal, expected_ordinal)
        << "CUDA plugin EP assigned ordinal " << actual_ordinal
        << " but cudaDeviceGetByPCIBusId(\"" << pci_bus_id << "\") resolved to "
        << expected_ordinal << ".";
  }

  EXPECT_GT(cuda_ep_device_count, 0)
      << "No CudaPluginExecutionProvider devices found after registration.";
}

// Verify that every CUDA device on the host has a corresponding CUDA plugin EP device
// with the correct pci_bus_id → ordinal mapping.
TEST_F(CudaPluginPciBusIdTest, AllCudaDevicesHaveMatchingEpDevices) {
  auto ep_devices = ort_env->GetEpDevices();

  // Collect the set of CUDA ordinals assigned by the plugin EP.
  std::vector<int> assigned_ordinals;
  for (const auto& ep_device : ep_devices) {
    if (strcmp(ep_device.EpName(), "CudaPluginExecutionProvider") != 0) {
      continue;
    }
    auto ep_metadata = ep_device.EpMetadata();
    const char* cuda_device_id_str = ep_metadata.GetValue("cuda_device_id");
    if (cuda_device_id_str != nullptr) {
      assigned_ordinals.push_back(std::stoi(cuda_device_id_str));
    }
  }

  // Every ordinal in [0, cuda_device_count_) should appear exactly once.
  std::sort(assigned_ordinals.begin(), assigned_ordinals.end());
  // Remove duplicates to check uniqueness after.
  auto unique_end = std::unique(assigned_ordinals.begin(), assigned_ordinals.end());

  EXPECT_EQ(std::distance(assigned_ordinals.begin(), unique_end),
            static_cast<ptrdiff_t>(assigned_ordinals.size()))
      << "Duplicate CUDA ordinals detected among plugin EP devices.";

  EXPECT_EQ(static_cast<int>(assigned_ordinals.size()), cuda_device_count_)
      << "Number of CUDA plugin EP devices (" << assigned_ordinals.size()
      << ") does not match CUDA device count (" << cuda_device_count_ << ").";
}

// Verify that cudaDeviceGetByPCIBusId round-trips with cudaDeviceGetPCIBusId
// for all CUDA devices. This validates the format consistency that the
// ORT device discovery code must produce.
TEST_F(CudaPluginPciBusIdTest, PciBusIdRoundTrip) {
  for (int ordinal = 0; ordinal < cuda_device_count_; ++ordinal) {
    // Get the PCI bus ID string from CUDA runtime for this ordinal.
    char pci_bus_id[64] = {};
    cudaError_t err = cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), ordinal);
    ASSERT_EQ(err, cudaSuccess)
        << "cudaDeviceGetPCIBusId failed for ordinal " << ordinal
        << ": " << cudaGetErrorString(err);

    // Round-trip: resolve back to ordinal.
    int resolved_ordinal = -1;
    err = cudaDeviceGetByPCIBusId(&resolved_ordinal, pci_bus_id);
    ASSERT_EQ(err, cudaSuccess)
        << "cudaDeviceGetByPCIBusId failed for pci_bus_id=\"" << pci_bus_id
        << "\": " << cudaGetErrorString(err);

    EXPECT_EQ(resolved_ordinal, ordinal)
        << "PCI bus ID round-trip failed: ordinal " << ordinal
        << " → pci_bus_id=\"" << pci_bus_id
        << "\" → resolved ordinal " << resolved_ordinal;
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)