// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/test_environment.h"
#include "test/util/include/api_asserts.h"

#include <vector>

using namespace onnxruntime::test;

namespace {
// Helper to get hardware devices using the two-step API pattern
void GetHardwareDevicesHelper(const OrtApi* api, OrtEnv* env,
                              std::vector<const OrtHardwareDevice*>& devices) {
  size_t num_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetNumHardwareDevices(env, &num_devices));
  devices.resize(num_devices);
  if (num_devices > 0) {
    ASSERT_ORTSTATUS_OK(api->GetHardwareDevices(env, devices.data(), num_devices));
  }
}
}  // namespace

// -----------------------------
// GetHardwareDeviceEpIncompatibilityDetails C API unit tests
// -----------------------------

TEST(GetHardwareDeviceEpIncompatibilityDetailsCapiTest, InvalidArguments_NullEnv) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  // Create env for GetHardwareDevices
  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  // Get a valid hardware device first
  std::vector<const OrtHardwareDevice*> hw_devices;
  ASSERT_NO_FATAL_FAILURE(GetHardwareDevicesHelper(api, env, hw_devices));
  ASSERT_GT(hw_devices.size(), 0u);

  // env == nullptr for GetHardwareDeviceEpIncompatibilityDetails
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  OrtStatus* st = api->GetHardwareDeviceEpIncompatibilityDetails(nullptr, "CPUExecutionProvider", hw_devices[0], &details);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEpIncompatibilityDetailsCapiTest, InvalidArguments_NullEpName) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  std::vector<const OrtHardwareDevice*> hw_devices;
  ASSERT_NO_FATAL_FAILURE(GetHardwareDevicesHelper(api, env, hw_devices));
  ASSERT_GT(hw_devices.size(), 0u);

  // ep_name == nullptr
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  OrtStatus* st = api->GetHardwareDeviceEpIncompatibilityDetails(env, nullptr, hw_devices[0], &details);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEpIncompatibilityDetailsCapiTest, InvalidArguments_EmptyEpName) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  std::vector<const OrtHardwareDevice*> hw_devices;
  ASSERT_NO_FATAL_FAILURE(GetHardwareDevicesHelper(api, env, hw_devices));
  ASSERT_GT(hw_devices.size(), 0u);

  // ep_name == ""
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  OrtStatus* st = api->GetHardwareDeviceEpIncompatibilityDetails(env, "", hw_devices[0], &details);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEpIncompatibilityDetailsCapiTest, InvalidArguments_NullHardwareDevice) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  // hw == nullptr
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  OrtStatus* st = api->GetHardwareDeviceEpIncompatibilityDetails(env, "CPUExecutionProvider", nullptr, &details);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEpIncompatibilityDetailsCapiTest, InvalidArguments_NullDetailsOutput) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  std::vector<const OrtHardwareDevice*> hw_devices;
  ASSERT_NO_FATAL_FAILURE(GetHardwareDevicesHelper(api, env, hw_devices));
  ASSERT_GT(hw_devices.size(), 0u);

  // details == nullptr
  OrtStatus* st = api->GetHardwareDeviceEpIncompatibilityDetails(env, "CPUExecutionProvider", hw_devices[0], nullptr);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEpIncompatibilityDetailsCapiTest, UnregisteredEp_ReturnsInvalidArgument) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  std::vector<const OrtHardwareDevice*> hw_devices;
  ASSERT_NO_FATAL_FAILURE(GetHardwareDevicesHelper(api, env, hw_devices));
  ASSERT_GT(hw_devices.size(), 0u);

  // Non-existent EP name should return INVALID_ARGUMENT
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  OrtStatus* st = api->GetHardwareDeviceEpIncompatibilityDetails(env, "NonExistentExecutionProvider", hw_devices[0], &details);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  EXPECT_THAT(api->GetErrorMessage(st), testing::HasSubstr("No valid factory found"));
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEpIncompatibilityDetailsCapiTest, CpuEp_ReturnsEmptyDetails) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  std::vector<const OrtHardwareDevice*> hw_devices;
  ASSERT_NO_FATAL_FAILURE(GetHardwareDevicesHelper(api, env, hw_devices));
  ASSERT_GT(hw_devices.size(), 0u);

  // CPU EP doesn't implement GetHardwareDeviceIncompatibilityDetails, so should return empty details
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api->GetHardwareDeviceEpIncompatibilityDetails(env, "CPUExecutionProvider", hw_devices[0], &details));
  ASSERT_NE(details, nullptr);

  // Verify empty details
  uint32_t reasons_bitmask = 0xFFFFFFFF;  // Initialize to non-zero to verify it gets set
  ASSERT_ORTSTATUS_OK(api->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, &reasons_bitmask));
  EXPECT_EQ(reasons_bitmask, 0u);

  int32_t error_code = -1;  // Initialize to non-zero to verify it gets set
  ASSERT_ORTSTATUS_OK(api->DeviceEpIncompatibilityDetails_GetErrorCode(details, &error_code));
  EXPECT_EQ(error_code, 0);

  const char* notes = reinterpret_cast<const char*>(0xDEADBEEF);  // Initialize to non-null
  ASSERT_ORTSTATUS_OK(api->DeviceEpIncompatibilityDetails_GetNotes(details, &notes));
  EXPECT_TRUE(notes == nullptr || strlen(notes) == 0);

  api->ReleaseDeviceEpIncompatibilityDetails(details);
  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEpIncompatibilityDetailsCapiTest, AccessorFunctions_NullDetails) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  // Test accessor functions with null details
  uint32_t reasons_bitmask = 0;
  OrtStatus* st = api->DeviceEpIncompatibilityDetails_GetReasonsBitmask(nullptr, &reasons_bitmask);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  int32_t error_code = 0;
  st = api->DeviceEpIncompatibilityDetails_GetErrorCode(nullptr, &error_code);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  const char* notes = nullptr;
  st = api->DeviceEpIncompatibilityDetails_GetNotes(nullptr, &notes);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);
}

TEST(GetHardwareDeviceEpIncompatibilityDetailsCapiTest, AccessorFunctions_NullOutputPtr) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  std::vector<const OrtHardwareDevice*> hw_devices;
  ASSERT_NO_FATAL_FAILURE(GetHardwareDevicesHelper(api, env, hw_devices));
  ASSERT_GT(hw_devices.size(), 0u);

  // Get a valid details object first
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api->GetHardwareDeviceEpIncompatibilityDetails(env, "CPUExecutionProvider", hw_devices[0], &details));
  ASSERT_NE(details, nullptr);

  // Test accessor functions with null output pointers
  OrtStatus* st = api->DeviceEpIncompatibilityDetails_GetReasonsBitmask(details, nullptr);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  st = api->DeviceEpIncompatibilityDetails_GetErrorCode(details, nullptr);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  st = api->DeviceEpIncompatibilityDetails_GetNotes(details, nullptr);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseDeviceEpIncompatibilityDetails(details);
  api->ReleaseEnv(env);
}

// -----------------------------
// GetNumHardwareDevices / GetHardwareDevices C API unit tests
// -----------------------------

TEST(GetHardwareDevicesCapiTest, GetNumHardwareDevices_InvalidArguments_NullEnv) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  size_t num_devices = 0;
  OrtStatus* st = api->GetNumHardwareDevices(nullptr, &num_devices);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);
}

TEST(GetHardwareDevicesCapiTest, GetNumHardwareDevices_InvalidArguments_NullNumDevices) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "HwDevicesTest", &env));
  EXPECT_NE(env, nullptr);

  OrtStatus* st = api->GetNumHardwareDevices(env, nullptr);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDevicesCapiTest, GetHardwareDevices_InvalidArguments_NullEnv) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  const OrtHardwareDevice* devices[1] = {nullptr};
  OrtStatus* st = api->GetHardwareDevices(nullptr, devices, 1);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);
}

TEST(GetHardwareDevicesCapiTest, GetHardwareDevices_InvalidArguments_NullDevices) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "HwDevicesTest", &env));
  EXPECT_NE(env, nullptr);

  OrtStatus* st = api->GetHardwareDevices(env, nullptr, 1);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDevicesCapiTest, GetHardwareDevices_InvalidArguments_ArrayTooSmall) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "HwDevicesTest", &env));
  EXPECT_NE(env, nullptr);

  // Get number of devices first
  size_t num_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetNumHardwareDevices(env, &num_devices));
  ASSERT_GT(num_devices, 0u);

  // Try to get devices with an undersized array (pass a valid pointer but claim size is 0)
  std::vector<const OrtHardwareDevice*> devices(1);                 // Allocate at least 1 element to avoid nullptr
  OrtStatus* st = api->GetHardwareDevices(env, devices.data(), 0);  // But claim size is 0
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  EXPECT_THAT(api->GetErrorMessage(st), testing::HasSubstr("num_devices is less than"));
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDevicesCapiTest, ReturnsDevices) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "HwDevicesTest", &env));
  EXPECT_NE(env, nullptr);

  // Get number of devices first
  size_t num_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetNumHardwareDevices(env, &num_devices));

  // Should return at least one device (CPU)
  EXPECT_GT(num_devices, 0u);

  // Allocate array and get devices
  std::vector<const OrtHardwareDevice*> devices(num_devices);
  ASSERT_ORTSTATUS_OK(api->GetHardwareDevices(env, devices.data(), num_devices));

  // Verify we can access device properties via C API accessor functions
  for (size_t i = 0; i < num_devices; ++i) {
    const OrtHardwareDevice* device = devices[i];
    ASSERT_NE(device, nullptr);
    // Device type should be valid (CPU, GPU, or NPU)
    OrtHardwareDeviceType device_type = api->HardwareDevice_Type(device);
    EXPECT_TRUE(device_type == OrtHardwareDeviceType_CPU ||
                device_type == OrtHardwareDeviceType_GPU ||
                device_type == OrtHardwareDeviceType_NPU);
    // Vendor should not be null
    const char* vendor = api->HardwareDevice_Vendor(device);
    EXPECT_NE(vendor, nullptr);
  }

  api->ReleaseEnv(env);
}
