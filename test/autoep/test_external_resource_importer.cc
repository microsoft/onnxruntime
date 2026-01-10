// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the External Resource Interop API using the example_plugin_ep.
// This tests the public ORT API without requiring actual D3D12/CUDA hardware.

#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"

#include "test/autoep/test_autoep_utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

class ExternalResourceImporterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Register the example EP and get the device using shared utility
    Utils::RegisterAndGetExampleEp(*ort_env, Utils::example_ep_info, registered_ep_);
    ASSERT_NE(registered_ep_.get(), nullptr) << "Example EP device not found";
    ep_device_ = registered_ep_.get();
  }

  const OrtInteropApi& GetInteropApi() const {
    return Ort::GetInteropApi();
  }

  RegisteredEpDeviceUniquePtr registered_ep_;
  const OrtEpDevice* ep_device_ = nullptr;
};

// Test: Create External Resource Importer
TEST_F(ExternalResourceImporterTest, CreateExternalResourceImporter) {
  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = GetInteropApi().CreateExternalResourceImporterForDevice(ep_device_, &importer);

  // Status should be nullptr on success (even if importer is null for unsupported EPs)
  ASSERT_EQ(status, nullptr) << "CreateExternalResourceImporterForDevice should succeed";

  // importer may be nullptr if EP doesn't support this optional feature
  if (importer == nullptr) {
    GTEST_SKIP() << "External resource interop not supported by this EP";
  }

  // Release the importer
  GetInteropApi().ReleaseExternalResourceImporter(importer);
}

// Test: Memory Import Capability
TEST_F(ExternalResourceImporterTest, CanImportMemory) {
  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = GetInteropApi().CreateExternalResourceImporterForDevice(ep_device_, &importer);
  ASSERT_EQ(status, nullptr) << "CreateExternalResourceImporterForDevice should succeed";
  if (importer == nullptr) {
    GTEST_SKIP() << "External resource interop not supported";
  }

  // Check D3D12 Resource support
  bool can_import_resource = false;
  status = GetInteropApi().CanImportMemory(
      importer, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE, &can_import_resource);
  ASSERT_EQ(status, nullptr) << "CanImportMemory for D3D12_RESOURCE should succeed";
  EXPECT_TRUE(can_import_resource) << "Example EP should support D3D12 Resource import";

  // Check D3D12 Heap support
  bool can_import_heap = false;
  status = GetInteropApi().CanImportMemory(
      importer, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP, &can_import_heap);
  ASSERT_EQ(status, nullptr) << "CanImportMemory for D3D12_HEAP should succeed";
  EXPECT_TRUE(can_import_heap) << "Example EP should support D3D12 Heap import";

  GetInteropApi().ReleaseExternalResourceImporter(importer);
}

// Test: Semaphore Import Capability
TEST_F(ExternalResourceImporterTest, CanImportSemaphore) {
  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = GetInteropApi().CreateExternalResourceImporterForDevice(ep_device_, &importer);
  ASSERT_EQ(status, nullptr) << "CreateExternalResourceImporterForDevice should succeed";
  if (importer == nullptr) {
    GTEST_SKIP() << "External resource interop not supported";
  }

  // Check D3D12 Fence support
  bool can_import_fence = false;
  status = GetInteropApi().CanImportSemaphore(
      importer, ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE, &can_import_fence);
  ASSERT_EQ(status, nullptr) << "CanImportSemaphore for D3D12_FENCE should succeed";
  EXPECT_TRUE(can_import_fence) << "Example EP should support D3D12 Fence import";

  GetInteropApi().ReleaseExternalResourceImporter(importer);
}

// Test: Import Memory (Simulated)
TEST_F(ExternalResourceImporterTest, ImportMemory) {
  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = GetInteropApi().CreateExternalResourceImporterForDevice(ep_device_, &importer);
  ASSERT_EQ(status, nullptr) << "CreateExternalResourceImporterForDevice should succeed";
  if (importer == nullptr) {
    GTEST_SKIP() << "External resource interop not supported";
  }

  // Import memory (using a dummy handle for testing)
  const size_t buffer_size = 1024 * sizeof(float);
  void* dummy_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(0x12345678));  // Simulated handle

  OrtExternalMemoryDescriptor mem_desc = {};
  mem_desc.version = ORT_API_VERSION;
  mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  mem_desc.native_handle = dummy_handle;
  mem_desc.size_bytes = buffer_size;
  mem_desc.offset_bytes = 0;

  OrtExternalMemoryHandle* mem_handle = nullptr;
  status = GetInteropApi().ImportMemory(importer, &mem_desc, &mem_handle);
  ASSERT_EQ(status, nullptr) << "ImportMemory should succeed";
  ASSERT_NE(mem_handle, nullptr) << "Memory handle should not be null";

  // Release memory handle
  GetInteropApi().ReleaseExternalMemoryHandle(mem_handle);

  GetInteropApi().ReleaseExternalResourceImporter(importer);
}

// Test: Create Tensor from Imported Memory
TEST_F(ExternalResourceImporterTest, CreateTensorFromMemory) {
  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = GetInteropApi().CreateExternalResourceImporterForDevice(ep_device_, &importer);
  ASSERT_EQ(status, nullptr) << "CreateExternalResourceImporterForDevice should succeed";
  if (importer == nullptr) {
    GTEST_SKIP() << "External resource interop not supported";
  }

  // Create tensor shape: [1, 3, 32, 32]
  const int64_t batch = 1, channels = 3, height = 32, width = 32;
  const int64_t shape[] = {batch, channels, height, width};
  const size_t num_elements = batch * channels * height * width;
  const size_t buffer_size = num_elements * sizeof(float);

  // Import memory
  void* dummy_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(0x12345678));

  OrtExternalMemoryDescriptor mem_desc = {};
  mem_desc.version = ORT_API_VERSION;
  mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  mem_desc.native_handle = dummy_handle;
  mem_desc.size_bytes = buffer_size;
  mem_desc.offset_bytes = 0;

  OrtExternalMemoryHandle* mem_handle = nullptr;
  status = GetInteropApi().ImportMemory(importer, &mem_desc, &mem_handle);
  ASSERT_EQ(status, nullptr);

  // Create tensor from imported memory
  OrtExternalTensorDescriptor tensor_desc = {};
  tensor_desc.version = ORT_API_VERSION;
  tensor_desc.element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  tensor_desc.shape = shape;
  tensor_desc.rank = 4;
  tensor_desc.offset_bytes = 0;

  OrtValue* tensor = nullptr;
  status = GetInteropApi().CreateTensorFromMemory(
      importer, mem_handle, &tensor_desc, &tensor);
  ASSERT_EQ(status, nullptr) << "CreateTensorFromMemory should succeed";
  ASSERT_NE(tensor, nullptr) << "Tensor should not be null";

  // Verify tensor properties
  OrtTensorTypeAndShapeInfo* type_info = nullptr;
  status = Ort::GetApi().GetTensorTypeAndShape(tensor, &type_info);
  ASSERT_EQ(status, nullptr);

  size_t rank = 0;
  status = Ort::GetApi().GetDimensionsCount(type_info, &rank);
  ASSERT_EQ(status, nullptr);
  EXPECT_EQ(rank, 4u);

  std::vector<int64_t> actual_shape(rank);
  status = Ort::GetApi().GetDimensions(type_info, actual_shape.data(), rank);
  ASSERT_EQ(status, nullptr);
  EXPECT_EQ(actual_shape[0], batch);
  EXPECT_EQ(actual_shape[1], channels);
  EXPECT_EQ(actual_shape[2], height);
  EXPECT_EQ(actual_shape[3], width);

  ONNXTensorElementDataType elem_type;
  status = Ort::GetApi().GetTensorElementType(type_info, &elem_type);
  ASSERT_EQ(status, nullptr);
  EXPECT_EQ(elem_type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  Ort::GetApi().ReleaseTensorTypeAndShapeInfo(type_info);

  // Cleanup
  Ort::GetApi().ReleaseValue(tensor);
  GetInteropApi().ReleaseExternalMemoryHandle(mem_handle);
  GetInteropApi().ReleaseExternalResourceImporter(importer);
}

// Test: Import Semaphore (Simulated)
TEST_F(ExternalResourceImporterTest, ImportSemaphore) {
  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = GetInteropApi().CreateExternalResourceImporterForDevice(ep_device_, &importer);
  ASSERT_EQ(status, nullptr) << "CreateExternalResourceImporterForDevice should succeed";
  if (importer == nullptr) {
    GTEST_SKIP() << "External resource interop not supported";
  }

  // Import semaphore (using a dummy handle for testing)
  void* dummy_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(0xABCDEF00));

  OrtExternalSemaphoreDescriptor sem_desc = {};
  sem_desc.version = ORT_API_VERSION;
  sem_desc.type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
  sem_desc.native_handle = dummy_handle;

  OrtExternalSemaphoreHandle* sem_handle = nullptr;
  status = GetInteropApi().ImportSemaphore(importer, &sem_desc, &sem_handle);
  ASSERT_EQ(status, nullptr) << "ImportSemaphore should succeed";
  ASSERT_NE(sem_handle, nullptr) << "Semaphore handle should not be null";

  // Release semaphore handle
  GetInteropApi().ReleaseExternalSemaphoreHandle(sem_handle);

  GetInteropApi().ReleaseExternalResourceImporter(importer);
}

// Test: Wait and Signal Semaphore (Simulated)
TEST_F(ExternalResourceImporterTest, WaitAndSignalSemaphore) {
  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = GetInteropApi().CreateExternalResourceImporterForDevice(ep_device_, &importer);
  ASSERT_EQ(status, nullptr) << "CreateExternalResourceImporterForDevice should succeed";
  if (importer == nullptr) {
    GTEST_SKIP() << "External resource interop not supported";
  }

  // Create a stream for the EP
  OrtSyncStream* stream = nullptr;
  status = Ort::GetApi().CreateSyncStreamForEpDevice(ep_device_, nullptr, &stream);
  ASSERT_EQ(status, nullptr) << "CreateSyncStreamForEpDevice should succeed";

  // Import semaphore
  void* dummy_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(0xABCDEF00));

  OrtExternalSemaphoreDescriptor sem_desc = {};
  sem_desc.version = ORT_API_VERSION;
  sem_desc.type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
  sem_desc.native_handle = dummy_handle;

  OrtExternalSemaphoreHandle* sem_handle = nullptr;
  status = GetInteropApi().ImportSemaphore(importer, &sem_desc, &sem_handle);
  ASSERT_EQ(status, nullptr);

  // Signal the semaphore with value 1
  status = GetInteropApi().SignalSemaphore(importer, sem_handle, stream, 1);
  ASSERT_EQ(status, nullptr) << "SignalSemaphore should succeed";

  // Wait for value 1 (should succeed immediately since we just signaled it)
  status = GetInteropApi().WaitSemaphore(importer, sem_handle, stream, 1);
  ASSERT_EQ(status, nullptr) << "WaitSemaphore should succeed";

  // Signal with value 5
  status = GetInteropApi().SignalSemaphore(importer, sem_handle, stream, 5);
  ASSERT_EQ(status, nullptr);

  // Wait for value 3 (should succeed since current value is 5)
  status = GetInteropApi().WaitSemaphore(importer, sem_handle, stream, 3);
  ASSERT_EQ(status, nullptr);

  // Cleanup
  GetInteropApi().ReleaseExternalSemaphoreHandle(sem_handle);
  Ort::GetApi().ReleaseSyncStream(stream);
  GetInteropApi().ReleaseExternalResourceImporter(importer);
}

// Test: Multiple Memory Imports
TEST_F(ExternalResourceImporterTest, MultipleMemoryImports) {
  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = GetInteropApi().CreateExternalResourceImporterForDevice(ep_device_, &importer);
  ASSERT_EQ(status, nullptr) << "CreateExternalResourceImporterForDevice should succeed";
  if (importer == nullptr) {
    GTEST_SKIP() << "External resource interop not supported";
  }

  constexpr int kNumBuffers = 5;
  std::vector<OrtExternalMemoryHandle*> handles(kNumBuffers);

  // Import multiple memory regions
  for (int i = 0; i < kNumBuffers; ++i) {
    OrtExternalMemoryDescriptor mem_desc = {};
    mem_desc.version = ORT_API_VERSION;
    mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    mem_desc.native_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(0x10000000 + i * 0x1000));
    mem_desc.size_bytes = (i + 1) * 1024;
    mem_desc.offset_bytes = 0;

    status = GetInteropApi().ImportMemory(importer, &mem_desc, &handles[i]);
    ASSERT_EQ(status, nullptr) << "ImportMemory " << i << " should succeed";
    ASSERT_NE(handles[i], nullptr);
  }

  // Release all handles
  for (int i = 0; i < kNumBuffers; ++i) {
    GetInteropApi().ReleaseExternalMemoryHandle(handles[i]);
  }

  GetInteropApi().ReleaseExternalResourceImporter(importer);
}

// Test: SessionGetEpDeviceForOutputs
TEST_F(ExternalResourceImporterTest, SessionGetEpDeviceForOutputs) {
  // Load a simple model with the example EP
  Ort::SessionOptions session_options;

  // Add the example EP to the session
  const OrtEpDevice* devices[] = {ep_device_};
  OrtStatus* status = Ort::GetApi().SessionOptionsAppendExecutionProvider_V2(
      session_options, *ort_env, devices, 1, nullptr, nullptr, 0);
  if (status != nullptr) {
    std::string error = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    GTEST_SKIP() << "Example EP not available: " << error;
  }

  // Create session with test model (mul_1.onnx - a simple model the example EP supports)
  Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), session_options);

  // Get output count
  size_t num_outputs = session.GetOutputCount();
  ASSERT_GT(num_outputs, 0U) << "Model should have at least one output";

  // Get EP devices for outputs
  std::vector<const OrtEpDevice*> output_devices(num_outputs);
  status = Ort::GetApi().SessionGetEpDeviceForOutputs(
      session, output_devices.data(), num_outputs);
  ASSERT_EQ(status, nullptr) << "SessionGetEpDeviceForOutputs should succeed";

  // Validate that we got EP devices (may be nullptr if not assigned to EP)
  // At least verify the call succeeded and returned valid array
  for (size_t i = 0; i < num_outputs; ++i) {
    if (output_devices[i] != nullptr) {
      // If an EP device is returned, validate it has a name
      const char* ep_name = Ort::GetApi().EpDevice_EpName(output_devices[i]);
      ASSERT_NE(ep_name, nullptr) << "EP device should have a name";
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
