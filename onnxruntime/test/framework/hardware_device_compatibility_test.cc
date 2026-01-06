// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/test_environment.h"
#include "test/util/include/api_asserts.h"

using namespace onnxruntime::test;

// -----------------------------
// GetHardwareDeviceEPIncompatibilityReasons C API unit tests
// -----------------------------

TEST(GetHardwareDeviceEPIncompatibilityReasonsCapiTest, InvalidArguments_NullEnv) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  // Create env for GetOrtHardwareDevices
  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  // Get a valid hardware device first
  const OrtHardwareDevice* const* hw_devices = nullptr;
  size_t num_hw_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetOrtHardwareDevices(env, &hw_devices, &num_hw_devices));
  ASSERT_GT(num_hw_devices, 0u);
  ASSERT_NE(hw_devices, nullptr);

  // env == nullptr for GetHardwareDeviceEPIncompatibilityReasons
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  OrtStatus* st = api->GetHardwareDeviceEPIncompatibilityReasons(nullptr, "CPUExecutionProvider", hw_devices[0], &details);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEPIncompatibilityReasonsCapiTest, InvalidArguments_NullEpName) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  const OrtHardwareDevice* const* hw_devices = nullptr;
  size_t num_hw_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetOrtHardwareDevices(env, &hw_devices, &num_hw_devices));
  ASSERT_GT(num_hw_devices, 0u);

  // ep_name == nullptr
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  OrtStatus* st = api->GetHardwareDeviceEPIncompatibilityReasons(env, nullptr, hw_devices[0], &details);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEPIncompatibilityReasonsCapiTest, InvalidArguments_EmptyEpName) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  const OrtHardwareDevice* const* hw_devices = nullptr;
  size_t num_hw_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetOrtHardwareDevices(env, &hw_devices, &num_hw_devices));
  ASSERT_GT(num_hw_devices, 0u);

  // ep_name == ""
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  OrtStatus* st = api->GetHardwareDeviceEPIncompatibilityReasons(env, "", hw_devices[0], &details);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEPIncompatibilityReasonsCapiTest, InvalidArguments_NullHardwareDevice) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  // hw == nullptr
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  OrtStatus* st = api->GetHardwareDeviceEPIncompatibilityReasons(env, "CPUExecutionProvider", nullptr, &details);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEPIncompatibilityReasonsCapiTest, InvalidArguments_NullDetailsOutput) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  const OrtHardwareDevice* const* hw_devices = nullptr;
  size_t num_hw_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetOrtHardwareDevices(env, &hw_devices, &num_hw_devices));
  ASSERT_GT(num_hw_devices, 0u);

  // details == nullptr
  OrtStatus* st = api->GetHardwareDeviceEPIncompatibilityReasons(env, "CPUExecutionProvider", hw_devices[0], nullptr);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEPIncompatibilityReasonsCapiTest, UnregisteredEp_ReturnsInvalidArgument) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  const OrtHardwareDevice* const* hw_devices = nullptr;
  size_t num_hw_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetOrtHardwareDevices(env, &hw_devices, &num_hw_devices));
  ASSERT_GT(num_hw_devices, 0u);

  // Non-existent EP name should return INVALID_ARGUMENT
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  OrtStatus* st = api->GetHardwareDeviceEPIncompatibilityReasons(env, "NonExistentExecutionProvider", hw_devices[0], &details);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  EXPECT_THAT(api->GetErrorMessage(st), testing::HasSubstr("No valid factory found"));
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetHardwareDeviceEPIncompatibilityReasonsCapiTest, CpuEp_ReturnsEmptyDetails) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  const OrtHardwareDevice* const* hw_devices = nullptr;
  size_t num_hw_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetOrtHardwareDevices(env, &hw_devices, &num_hw_devices));
  ASSERT_GT(num_hw_devices, 0u);

  // CPU EP doesn't implement GetHardwareDeviceIncompatibilityReasons, so should return empty details
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api->GetHardwareDeviceEPIncompatibilityReasons(env, "CPUExecutionProvider", hw_devices[0], &details));
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

TEST(GetHardwareDeviceEPIncompatibilityReasonsCapiTest, AccessorFunctions_NullDetails) {
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

TEST(GetHardwareDeviceEPIncompatibilityReasonsCapiTest, AccessorFunctions_NullOutputPtr) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EpIncompatTest", &env));
  EXPECT_NE(env, nullptr);

  const OrtHardwareDevice* const* hw_devices = nullptr;
  size_t num_hw_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetOrtHardwareDevices(env, &hw_devices, &num_hw_devices));
  ASSERT_GT(num_hw_devices, 0u);

  // Get a valid details object first
  OrtDeviceEpIncompatibilityDetails* details = nullptr;
  ASSERT_ORTSTATUS_OK(api->GetHardwareDeviceEPIncompatibilityReasons(env, "CPUExecutionProvider", hw_devices[0], &details));
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
// GetOrtHardwareDevices C API unit tests
// -----------------------------

TEST(GetOrtHardwareDevicesCapiTest, InvalidArguments_NullEnv) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  const OrtHardwareDevice* const* devices = nullptr;
  size_t num_devices = 0;
  OrtStatus* st = api->GetOrtHardwareDevices(nullptr, &devices, &num_devices);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);
}

TEST(GetOrtHardwareDevicesCapiTest, InvalidArguments_NullDevices) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "HwDevicesTest", &env));
  EXPECT_NE(env, nullptr);

  size_t num_devices = 0;
  OrtStatus* st = api->GetOrtHardwareDevices(env, nullptr, &num_devices);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetOrtHardwareDevicesCapiTest, InvalidArguments_NullNumDevices) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "HwDevicesTest", &env));
  EXPECT_NE(env, nullptr);

  const OrtHardwareDevice* const* devices = nullptr;
  OrtStatus* st = api->GetOrtHardwareDevices(env, &devices, nullptr);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(api->GetErrorCode(st), ORT_INVALID_ARGUMENT);
  api->ReleaseStatus(st);

  api->ReleaseEnv(env);
}

TEST(GetOrtHardwareDevicesCapiTest, ReturnsDevices) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);

  OrtEnv* env = nullptr;
  EXPECT_EQ(nullptr, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "HwDevicesTest", &env));
  EXPECT_NE(env, nullptr);

  const OrtHardwareDevice* const* devices = nullptr;
  size_t num_devices = 0;
  ASSERT_ORTSTATUS_OK(api->GetOrtHardwareDevices(env, &devices, &num_devices));

  // Should return at least one device (CPU)
  EXPECT_GT(num_devices, 0u);
  EXPECT_NE(devices, nullptr);

  // Verify we can access device properties
  for (size_t i = 0; i < num_devices; ++i) {
    const OrtHardwareDevice* device = devices[i];
    // Device type should be valid (CPU, GPU, or NPU)
    EXPECT_TRUE(device->type == OrtHardwareDeviceType_CPU ||
                device->type == OrtHardwareDeviceType_GPU ||
                device->type == OrtHardwareDeviceType_NPU);
    // Vendor should not be empty
    EXPECT_FALSE(device->vendor.empty());
  }

  api->ReleaseEnv(env);
}
