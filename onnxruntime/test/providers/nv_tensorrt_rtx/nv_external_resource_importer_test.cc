// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This test validates the D3D12 ↔ CUDA external resource import functionality
// for the NvTensorRtRtx execution provider.
//
// Test Coverage:
// 1. External Resource Importer creation and destruction
// 2. Memory import capability check (D3D12 Resource & Heap)
// 3. Semaphore import capability check (D3D12 Fence)
// 4. D3D12 shared resource import to CUDA
// 5. Tensor creation from imported external memory
// 6. D3D12 timeline fence import for GPU synchronization
// 7. Wait/Signal semaphore operations
// 8. Full inference pipeline with zero-copy external memory
// 9. Full inference pipeline with zero-copy external memory and a CIG context

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/common/random_generator.h"

#include <thread>
#include <chrono>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cstdio>

#if defined(_WIN32)
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
#endif

// Include CUDA headers for pointer attribute verification
#include <cuda.h>
#include <cuda_runtime.h>
#include <core/providers/nv_tensorrt_rtx/nv_provider_options.h>

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(_WIN32)

// Dynamic CUDA driver function loader
class CudaDriverLoader {
 private:
  HMODULE cuda_driver_dll_ = nullptr;

  // CUDA Driver API function pointers
  using cuCtxCreate_v4_t = CUresult (*)(CUcontext*, CUctxCreateParams*, unsigned int, CUdevice);
  using cuCtxDestroy_t = CUresult (*)(CUcontext);
  using cuCtxGetCurrent_t = CUresult (*)(CUcontext*);
  using cuCtxSetCurrent_t = CUresult (*)(CUcontext);

 public:
  cuCtxCreate_v4_t cuCtxCreate_v4_fn = nullptr;
  cuCtxDestroy_t cuCtxDestroy_fn = nullptr;
  cuCtxSetCurrent_t cuCtxSetCurrent_fn = nullptr;
  cuCtxGetCurrent_t cuCtxGetCurrent_fn = nullptr;

  CudaDriverLoader() {
    // Load CUDA driver library dynamically
    cuda_driver_dll_ = LoadLibraryA("nvcuda.dll");
    if (cuda_driver_dll_) {
      cuCtxCreate_v4_fn = reinterpret_cast<cuCtxCreate_v4_t>(
          GetProcAddress(cuda_driver_dll_, "cuCtxCreate_v4"));
      cuCtxDestroy_fn = reinterpret_cast<cuCtxDestroy_t>(
          GetProcAddress(cuda_driver_dll_, "cuCtxDestroy"));
      cuCtxSetCurrent_fn = reinterpret_cast<cuCtxSetCurrent_t>(
          GetProcAddress(cuda_driver_dll_, "cuCtxSetCurrent"));
      cuCtxGetCurrent_fn = reinterpret_cast<cuCtxGetCurrent_t>(
          GetProcAddress(cuda_driver_dll_, "cuCtxGetCurrent"));
    }
  }

  ~CudaDriverLoader() {
    if (cuda_driver_dll_) {
      FreeLibrary(cuda_driver_dll_);
    }
  }

  bool IsLoaded() const {
    return cuda_driver_dll_ != nullptr &&
           cuCtxCreate_v4_fn != nullptr &&
           cuCtxSetCurrent_fn != nullptr &&
           cuCtxDestroy_fn != nullptr &&
           cuCtxGetCurrent_fn != nullptr;
  }
};

// Helper functions for D3D12 resource creation
class D3D12ResourceHelper {
 public:
  static void CreateSharedBuffer(ID3D12Device* device,
                                 size_t size,
                                 ID3D12Resource** out_resource,
                                 D3D12_RESOURCE_STATES initial_state = D3D12_RESOURCE_STATE_COMMON) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES heap_props = {};
    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    // Create with SHARED heap flag for cross-API import
    HRESULT hr = device->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_SHARED,
        &desc,
        initial_state,
        nullptr,
        IID_PPV_ARGS(out_resource));

    if (FAILED(hr)) {
      GTEST_FAIL() << "Failed to create shared D3D12 buffer, HRESULT: 0x" << std::hex << hr;
    }
  }

  static void CreateUploadBuffer(ID3D12Device* device, size_t size, ID3D12Resource** out_resource) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES heap_props = {};
    heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;

    HRESULT hr = device->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(out_resource));

    if (FAILED(hr)) {
      GTEST_FAIL() << "Failed to create upload buffer, HRESULT: 0x" << std::hex << hr;
    }
  }

  static void CreateReadbackBuffer(ID3D12Device* device, size_t size, ID3D12Resource** out_resource) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    D3D12_HEAP_PROPERTIES heap_props = {};
    heap_props.Type = D3D12_HEAP_TYPE_READBACK;

    HRESULT hr = device->CreateCommittedResource(
        &heap_props,
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(out_resource));

    if (FAILED(hr)) {
      GTEST_FAIL() << "Failed to create readback buffer, HRESULT: 0x" << std::hex << hr;
    }
  }

  static void FlushAndWait(ID3D12Device* device, ID3D12CommandQueue* queue) {
    ComPtr<ID3D12Fence> fence;
    HRESULT hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
      GTEST_FAIL() << "Failed to create fence for flush, HRESULT: 0x" << std::hex << hr;
    }

    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    queue->Signal(fence.Get(), 1);
    fence->SetEventOnCompletion(1, event);
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
  }
};

// Test Fixture
class NvExecutionProviderExternalResourceImporterTest : public testing::Test {
 protected:
  void SetUp() override {
    // Get the ORT API
    ort_api_ = &Ort::GetApi();
    ort_interop_api_ = &Ort::GetInteropApi();

    // Try to create D3D12 device
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12_device_));
    if (FAILED(hr)) {
      d3d12_available_ = false;
      return;
    }
    d3d12_available_ = true;

    // Create command queue
    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    hr = d3d12_device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&command_queue_));
    if (FAILED(hr)) {
      d3d12_available_ = false;
      return;
    }

    // Create command allocator and list
    hr = d3d12_device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                               IID_PPV_ARGS(&command_allocator_));
    if (FAILED(hr)) {
      d3d12_available_ = false;
      return;
    }

    hr = d3d12_device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                          command_allocator_.Get(), nullptr,
                                          IID_PPV_ARGS(&command_list_));
    if (FAILED(hr)) {
      d3d12_available_ = false;
      return;
    }
    command_list_->Close();
    // Register NvTensorRtRtx EP
    Utils::RegisterAndGetNvTensorRtRtxEp(*ort_env, registered_ep_);
    if (registered_ep_.get() == nullptr) {
      ep_available_ = false;
      return;
    }
    ep_available_ = true;
    ep_device_ = registered_ep_.get();
  }

  void TearDown() override {
    // Release resources
    command_list_.Reset();
    command_allocator_.Reset();
    command_queue_.Reset();
    d3d12_device_.Reset();
  }

  bool IsD3D12Available() const { return d3d12_available_; }
  bool IsEPAvailable() const { return ep_available_; }

  ComPtr<ID3D12Device> d3d12_device_;
  ComPtr<ID3D12CommandQueue> command_queue_;
  ComPtr<ID3D12CommandAllocator> command_allocator_;
  ComPtr<ID3D12GraphicsCommandList> command_list_;
  const OrtApi* ort_api_ = nullptr;
  const OrtInteropApi* ort_interop_api_ = nullptr;
  RegisteredEpDeviceUniquePtr registered_ep_;  // RAII - auto-unregisters EP
  const OrtEpDevice* ep_device_ = nullptr;
  bool d3d12_available_ = false;
  bool ep_available_ = false;
};

// Test: External Resource Importer Creation
TEST_F(NvExecutionProviderExternalResourceImporterTest, CreateExternalResourceImporter) {
  if (!IsD3D12Available()) {
    GTEST_SKIP() << "D3D12 not available";
  }
  if (!IsEPAvailable()) {
    GTEST_SKIP() << "NvTensorRtRtx EP not available";
  }

  // Create external resource importer
  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);

  if (status != nullptr) {
    std::string error = ort_api_->GetErrorMessage(status);
    ort_api_->ReleaseStatus(status);
    GTEST_SKIP() << "CreateExternalResourceImporterForDevice not supported: " << error;
  }

  if (importer == nullptr) {
    // EP doesn't support external resource import yet
    GTEST_SKIP() << "External resource import not yet implemented by this EP";
  }

  // Release the importer
  ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Memory Import Capability Check
TEST_F(NvExecutionProviderExternalResourceImporterTest, CanImportMemoryCapabilities) {
  if (!IsD3D12Available()) {
    GTEST_SKIP() << "D3D12 not available";
  }
  if (!IsEPAvailable()) {
    GTEST_SKIP() << "NvTensorRtRtx EP not available";
  }

  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
  if (status != nullptr || importer == nullptr) {
    if (status != nullptr) ort_api_->ReleaseStatus(status);
    GTEST_SKIP() << "External resource import not supported";
  }

  // Check D3D12 Resource support
  bool can_import_resource = false;
  status = ort_interop_api_->CanImportMemory(
      importer, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE, &can_import_resource);
  ASSERT_EQ(status, nullptr) << "CanImportMemory for D3D12_RESOURCE should succeed";
  EXPECT_TRUE(can_import_resource) << "Should support D3D12 Resource import";

  // Check D3D12 Heap support
  bool can_import_heap = false;
  status = ort_interop_api_->CanImportMemory(
      importer, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP, &can_import_heap);
  ASSERT_EQ(status, nullptr) << "CanImportMemory for D3D12_HEAP should succeed";
  EXPECT_TRUE(can_import_heap) << "Should support D3D12 Heap import";

  ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Semaphore Import Capability Check
TEST_F(NvExecutionProviderExternalResourceImporterTest, CanImportSemaphoreCapabilities) {
  if (!IsD3D12Available()) {
    GTEST_SKIP() << "D3D12 not available";
  }
  if (!IsEPAvailable()) {
    GTEST_SKIP() << "NvTensorRtRtx EP not available";
  }

  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
  if (status != nullptr || importer == nullptr) {
    if (status != nullptr) ort_api_->ReleaseStatus(status);
    GTEST_SKIP() << "External resource import not supported";
  }

  // Check D3D12 Fence support
  bool can_import_fence = false;
  status = ort_interop_api_->CanImportSemaphore(
      importer, ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE, &can_import_fence);
  ASSERT_EQ(status, nullptr) << "CanImportSemaphore for D3D12_FENCE should succeed";
  EXPECT_TRUE(can_import_fence) << "Should support D3D12 Fence import";

  ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Import D3D12 Shared Resource
TEST_F(NvExecutionProviderExternalResourceImporterTest, ImportD3D12SharedResource) {
  if (!IsD3D12Available()) {
    GTEST_SKIP() << "D3D12 not available";
  }
  if (!IsEPAvailable()) {
    GTEST_SKIP() << "NvTensorRtRtx EP not available";
  }

  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
  if (status != nullptr || importer == nullptr) {
    if (status != nullptr) ort_api_->ReleaseStatus(status);
    GTEST_SKIP() << "External resource import not supported";
  }

  // Create a shared D3D12 buffer
  const size_t buffer_size = 1024 * sizeof(float);
  ComPtr<ID3D12Resource> d3d12_buffer;
  D3D12ResourceHelper::CreateSharedBuffer(d3d12_device_.Get(), buffer_size, &d3d12_buffer);

  // Create shared handle
  HANDLE shared_handle = nullptr;
  HRESULT hr = d3d12_device_->CreateSharedHandle(d3d12_buffer.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle);
  ASSERT_TRUE(SUCCEEDED(hr)) << "Failed to create shared handle";

  // Import the memory
  OrtExternalMemoryDescriptor mem_desc = {};
  mem_desc.version = ORT_API_VERSION;
  mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  mem_desc.native_handle = shared_handle;
  mem_desc.size_bytes = buffer_size;
  mem_desc.offset_bytes = 0;

  OrtExternalMemoryHandle* mem_handle = nullptr;
  status = ort_interop_api_->ImportMemory(importer, &mem_desc, &mem_handle);
  ASSERT_EQ(status, nullptr) << "ImportMemory should succeed";
  ASSERT_NE(mem_handle, nullptr) << "Memory handle should not be null";

  // Release memory handle
  ort_interop_api_->ReleaseExternalMemoryHandle(mem_handle);

  // Close shared handle
  CloseHandle(shared_handle);

  ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Create Tensor from Imported Memory
TEST_F(NvExecutionProviderExternalResourceImporterTest, CreateTensorFromImportedMemory) {
  if (!IsD3D12Available()) {
    GTEST_SKIP() << "D3D12 not available";
  }
  if (!IsEPAvailable()) {
    GTEST_SKIP() << "NvTensorRtRtx EP not available";
  }

  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
  if (status != nullptr || importer == nullptr) {
    if (status != nullptr) ort_api_->ReleaseStatus(status);
    GTEST_SKIP() << "External resource import not supported";
  }

  // Create tensor shape: [1, 3, 32, 32] (batch, channels, height, width)
  const int64_t batch = 1, channels = 3, height = 32, width = 32;
  const int64_t shape[] = {batch, channels, height, width};
  const size_t num_elements = batch * channels * height * width;
  const size_t buffer_size = num_elements * sizeof(float);

  // Create shared D3D12 buffer
  ComPtr<ID3D12Resource> d3d12_buffer;
  D3D12ResourceHelper::CreateSharedBuffer(d3d12_device_.Get(), buffer_size, &d3d12_buffer);

  // Create shared handle
  HANDLE shared_handle = nullptr;
  HRESULT hr = d3d12_device_->CreateSharedHandle(d3d12_buffer.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle);
  ASSERT_TRUE(SUCCEEDED(hr));

  // Import the memory
  OrtExternalMemoryDescriptor mem_desc = {};
  mem_desc.version = ORT_API_VERSION;
  mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  mem_desc.native_handle = shared_handle;
  mem_desc.size_bytes = buffer_size;
  mem_desc.offset_bytes = 0;

  OrtExternalMemoryHandle* mem_handle = nullptr;
  status = ort_interop_api_->ImportMemory(importer, &mem_desc, &mem_handle);
  ASSERT_EQ(status, nullptr);

  // Create tensor from imported memory
  OrtExternalTensorDescriptor tensor_desc = {};
  tensor_desc.version = ORT_API_VERSION;
  tensor_desc.element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  tensor_desc.shape = shape;
  tensor_desc.rank = 4;
  tensor_desc.offset_bytes = 0;

  OrtValue* tensor = nullptr;
  status = ort_interop_api_->CreateTensorFromMemory(importer, mem_handle, &tensor_desc, &tensor);
  ASSERT_EQ(status, nullptr) << "CreateTensorFromMemory should succeed";
  ASSERT_NE(tensor, nullptr) << "Tensor should not be null";

  // Verify tensor properties
  OrtTensorTypeAndShapeInfo* type_info = nullptr;
  status = ort_api_->GetTensorTypeAndShape(tensor, &type_info);
  ASSERT_EQ(status, nullptr);

  size_t rank = 0;
  ort_api_->GetDimensionsCount(type_info, &rank);
  EXPECT_EQ(rank, 4u);

  std::vector<int64_t> actual_shape(rank);
  ort_api_->GetDimensions(type_info, actual_shape.data(), rank);
  EXPECT_EQ(actual_shape[0], batch);
  EXPECT_EQ(actual_shape[1], channels);
  EXPECT_EQ(actual_shape[2], height);
  EXPECT_EQ(actual_shape[3], width);

  ONNXTensorElementDataType elem_type;
  ort_api_->GetTensorElementType(type_info, &elem_type);
  EXPECT_EQ(elem_type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  ort_api_->ReleaseTensorTypeAndShapeInfo(type_info);

  // Get the tensor's data pointer and verify it's CUDA device memory
  // This proves the D3D12 to CUDA memory import actually happened
  void* tensor_data = nullptr;
  status = ort_api_->GetTensorMutableData(tensor, &tensor_data);
  ASSERT_EQ(status, nullptr) << "GetTensorMutableData should succeed";
  ASSERT_NE(tensor_data, nullptr) << "Tensor data pointer should not be null";

  // Use cudaPointerGetAttributes to verify this is CUDA device memory
  cudaPointerAttributes attrs;
  cudaError_t cuda_err = cudaPointerGetAttributes(&attrs, tensor_data);
  ASSERT_EQ(cuda_err, cudaSuccess) << "cudaPointerGetAttributes failed: " << cudaGetErrorString(cuda_err);
  EXPECT_EQ(attrs.type, cudaMemoryTypeDevice)
      << "Memory should be CUDA device memory, but got type " << attrs.type
      << " (cudaMemoryTypeDevice=" << cudaMemoryTypeDevice << ")";
  EXPECT_NE(attrs.device, -1) << "Device should be valid";

  // Cleanup
  ort_api_->ReleaseValue(tensor);
  ort_interop_api_->ReleaseExternalMemoryHandle(mem_handle);
  CloseHandle(shared_handle);
  ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Import D3D12 Timeline Fence
TEST_F(NvExecutionProviderExternalResourceImporterTest, ImportD3D12Fence) {
  if (!IsD3D12Available()) {
    GTEST_SKIP() << "D3D12 not available";
  }
  if (!IsEPAvailable()) {
    GTEST_SKIP() << "NvTensorRtRtx EP not available";
  }

  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
  if (status != nullptr || importer == nullptr) {
    if (status != nullptr) ort_api_->ReleaseStatus(status);
    GTEST_SKIP() << "External resource import not supported";
  }

  // Create a D3D12 fence with SHARED flag for cross-API import
  ComPtr<ID3D12Fence> d3d12_fence;
  HRESULT hr = d3d12_device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&d3d12_fence));
  ASSERT_TRUE(SUCCEEDED(hr)) << "Failed to create D3D12 fence";

  // Create shared handle
  HANDLE shared_handle = nullptr;
  hr = d3d12_device_->CreateSharedHandle(d3d12_fence.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle);
  ASSERT_TRUE(SUCCEEDED(hr)) << "Failed to create shared fence handle";

  // Import the semaphore
  OrtExternalSemaphoreDescriptor sem_desc = {};
  sem_desc.version = ORT_API_VERSION;
  sem_desc.type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
  sem_desc.native_handle = shared_handle;

  OrtExternalSemaphoreHandle* sem_handle = nullptr;
  status = ort_interop_api_->ImportSemaphore(importer, &sem_desc, &sem_handle);
  ASSERT_EQ(status, nullptr) << "ImportSemaphore should succeed";
  ASSERT_NE(sem_handle, nullptr) << "Semaphore handle should not be null";

  // Release semaphore handle
  ort_interop_api_->ReleaseExternalSemaphoreHandle(sem_handle);

  // Close shared handle
  CloseHandle(shared_handle);

  ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Wait and Signal Semaphore
TEST_F(NvExecutionProviderExternalResourceImporterTest, WaitAndSignalSemaphore) {
  if (!IsD3D12Available()) {
    GTEST_SKIP() << "D3D12 not available";
  }
  if (!IsEPAvailable()) {
    GTEST_SKIP() << "NvTensorRtRtx EP not available";
  }

  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
  if (status != nullptr || importer == nullptr) {
    if (status != nullptr) ort_api_->ReleaseStatus(status);
    GTEST_SKIP() << "External resource import not supported";
  }

  // Create a CUDA stream via ORT
  OrtSyncStream* ort_stream = nullptr;
  status = ort_api_->CreateSyncStreamForEpDevice(ep_device_, nullptr, &ort_stream);
  ASSERT_EQ(status, nullptr) << "CreateSyncStreamForEpDevice should succeed";

  // Create a D3D12 fence
  ComPtr<ID3D12Fence> d3d12_fence;
  HRESULT hr = d3d12_device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&d3d12_fence));
  ASSERT_TRUE(SUCCEEDED(hr));

  HANDLE shared_handle = nullptr;
  hr = d3d12_device_->CreateSharedHandle(d3d12_fence.Get(), nullptr, GENERIC_ALL, nullptr, &shared_handle);
  ASSERT_TRUE(SUCCEEDED(hr));

  // Import semaphore
  OrtExternalSemaphoreDescriptor sem_desc = {};
  sem_desc.version = ORT_API_VERSION;
  sem_desc.type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
  sem_desc.native_handle = shared_handle;

  OrtExternalSemaphoreHandle* sem_handle = nullptr;
  status = ort_interop_api_->ImportSemaphore(importer, &sem_desc, &sem_handle);
  ASSERT_EQ(status, nullptr);

  // Signal the fence from D3D12 side
  uint64_t signal_value = 1;
  command_queue_->Signal(d3d12_fence.Get(), signal_value);

  // Wait on the fence from CUDA side
  status = ort_interop_api_->WaitSemaphore(importer, sem_handle, ort_stream, signal_value);
  ASSERT_EQ(status, nullptr) << "WaitSemaphore should succeed";

  // Signal from CUDA side
  uint64_t cuda_signal_value = 2;
  status = ort_interop_api_->SignalSemaphore(importer, sem_handle, ort_stream, cuda_signal_value);
  ASSERT_EQ(status, nullptr) << "SignalSemaphore should succeed";

  // Synchronize by getting the native stream handle and calling cudaStreamSynchronize
  // (In real code, the signal enqueued on the stream will complete when the stream is flushed)
  void* stream_handle = ort_api_->SyncStream_GetHandle(ort_stream);
  ASSERT_NE(stream_handle, nullptr);
  // Note: In production code, you'd call cudaStreamSynchronize((cudaStream_t)stream_handle)
  // For this test, we just verify the handle is valid

  // Verify D3D12 can see the signaled value
  HANDLE wait_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  d3d12_fence->SetEventOnCompletion(cuda_signal_value, wait_event);
  DWORD wait_result = WaitForSingleObject(wait_event, 5000);  // 5 second timeout
  CloseHandle(wait_event);
  EXPECT_EQ(wait_result, WAIT_OBJECT_0) << "D3D12 should see the fence signaled by CUDA";

  // Cleanup
  ort_interop_api_->ReleaseExternalSemaphoreHandle(sem_handle);
  CloseHandle(shared_handle);
  ort_api_->ReleaseSyncStream(ort_stream);
  ort_interop_api_->ReleaseExternalResourceImporter(importer);
}

// Test: Full Inference with External Memory (E2E)
// This test validates the complete D3D12 to CUDA interop pipeline:
// 1. Create D3D12 shared resources and fences
// 2. Import them into CUDA via OrtExternalResourceImporter
// 3. Create ORT tensors from imported memory
// 4. Run inference with proper synchronization
// 5. Verify output correctness
TEST_F(NvExecutionProviderExternalResourceImporterTest, FullInferenceWithExternalMemory) {
  if (!IsD3D12Available()) {
    GTEST_SKIP() << "D3D12 not available";
  }
  if (!IsEPAvailable()) {
    GTEST_SKIP() << "NvTensorRtRtx EP not available";
  }

  // Create a simple ReLU model using shared utility pattern
  PathString model_path = ORT_TSTR("external_mem_relu_test.onnx");
  {
    onnxruntime::Model model("relu_test", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    ONNX_NAMESPACE::TypeProto tensor_type;
    tensor_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
    tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(64);
    tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(64);

    auto& input_arg = graph.GetOrCreateNodeArg("X", &tensor_type);
    auto& output_arg = graph.GetOrCreateNodeArg("Y", &tensor_type);
    graph.AddNode("relu", "Relu", "ReLU operation", {&input_arg}, {&output_arg});

    ASSERT_STATUS_OK(graph.Resolve());
    ASSERT_STATUS_OK(onnxruntime::Model::Save(model, model_path));
  }

  const int64_t batch = 1, channels = 3, dim = 64;
  const int64_t shape[] = {batch, channels, dim, dim};
  const size_t num_elements = batch * channels * dim * dim;
  const size_t buffer_size = num_elements * sizeof(float);

  // Create external resource importer
  OrtExternalResourceImporter* importer = nullptr;
  OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
  if (status != nullptr || importer == nullptr) {
    if (status != nullptr) ort_api_->ReleaseStatus(status);
    clearFileIfExists(model_path);
    GTEST_SKIP() << "External resource import not supported";
  }

  // Create CUDA stream via ORT
  OrtSyncStream* ort_stream = nullptr;
  status = ort_api_->CreateSyncStreamForEpDevice(ep_device_, nullptr, &ort_stream);
  ASSERT_EQ(status, nullptr);

  // Create shared D3D12 buffers for input and output
  ComPtr<ID3D12Resource> input_buffer, output_buffer;
  D3D12ResourceHelper::CreateSharedBuffer(d3d12_device_.Get(), buffer_size, &input_buffer);
  D3D12ResourceHelper::CreateSharedBuffer(d3d12_device_.Get(), buffer_size, &output_buffer);

  // Create shared handles for cross-API import
  HANDLE input_handle = nullptr, output_handle = nullptr;
  d3d12_device_->CreateSharedHandle(input_buffer.Get(), nullptr, GENERIC_ALL, nullptr, &input_handle);
  d3d12_device_->CreateSharedHandle(output_buffer.Get(), nullptr, GENERIC_ALL, nullptr, &output_handle);

  // Import memory into CUDA
  OrtExternalMemoryDescriptor input_mem_desc = {};
  input_mem_desc.version = ORT_API_VERSION;
  input_mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  input_mem_desc.native_handle = input_handle;
  input_mem_desc.size_bytes = buffer_size;
  input_mem_desc.offset_bytes = 0;

  OrtExternalMemoryDescriptor output_mem_desc = input_mem_desc;
  output_mem_desc.native_handle = output_handle;

  OrtExternalMemoryHandle *input_mem = nullptr, *output_mem = nullptr;
  status = ort_interop_api_->ImportMemory(importer, &input_mem_desc, &input_mem);
  ASSERT_EQ(status, nullptr) << "ImportMemory for input should succeed (proves cuImportExternalMemory called)";
  status = ort_interop_api_->ImportMemory(importer, &output_mem_desc, &output_mem);
  ASSERT_EQ(status, nullptr) << "ImportMemory for output should succeed";

  // Create ORT tensors from imported memory
  OrtExternalTensorDescriptor tensor_desc = {};
  tensor_desc.version = ORT_API_VERSION;
  tensor_desc.element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  tensor_desc.shape = shape;
  tensor_desc.rank = 4;
  tensor_desc.offset_bytes = 0;

  OrtValue *input_tensor = nullptr, *output_tensor = nullptr;
  status = ort_interop_api_->CreateTensorFromMemory(importer, input_mem, &tensor_desc, &input_tensor);
  ASSERT_EQ(status, nullptr);
  status = ort_interop_api_->CreateTensorFromMemory(importer, output_mem, &tensor_desc, &output_tensor);
  ASSERT_EQ(status, nullptr);

  // Verify the tensor data pointers are CUDA device memory
  void* input_data_ptr = nullptr;
  void* output_data_ptr = nullptr;
  status = ort_api_->GetTensorMutableData(input_tensor, &input_data_ptr);
  ASSERT_EQ(status, nullptr);
  status = ort_api_->GetTensorMutableData(output_tensor, &output_data_ptr);
  ASSERT_EQ(status, nullptr);

  cudaPointerAttributes input_attrs, output_attrs;
  ASSERT_EQ(cudaPointerGetAttributes(&input_attrs, input_data_ptr), cudaSuccess);
  ASSERT_EQ(cudaPointerGetAttributes(&output_attrs, output_data_ptr), cudaSuccess);
  EXPECT_EQ(input_attrs.type, cudaMemoryTypeDevice) << "Input tensor must be CUDA device memory";
  EXPECT_EQ(output_attrs.type, cudaMemoryTypeDevice) << "Output tensor must be CUDA device memory";

  // Create D3D12 fence for bidirectional synchronization
  ComPtr<ID3D12Fence> sync_fence;
  d3d12_device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&sync_fence));
  HANDLE fence_handle = nullptr;
  d3d12_device_->CreateSharedHandle(sync_fence.Get(), nullptr, GENERIC_ALL, nullptr, &fence_handle);

  OrtExternalSemaphoreDescriptor sem_desc = {};
  sem_desc.version = ORT_API_VERSION;
  sem_desc.type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
  sem_desc.native_handle = fence_handle;

  OrtExternalSemaphoreHandle* sem_handle = nullptr;
  status = ort_interop_api_->ImportSemaphore(importer, &sem_desc, &sem_handle);
  ASSERT_EQ(status, nullptr) << "ImportSemaphore should succeed";

  // Setup test data via D3D12 upload buffer
  ComPtr<ID3D12Resource> upload_buffer;
  D3D12ResourceHelper::CreateUploadBuffer(d3d12_device_.Get(), buffer_size, &upload_buffer);

  // Generate test data: alternating positive and negative values for ReLU verification
  std::vector<float> test_data(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    test_data[i] = (i % 2 == 0) ? static_cast<float>(i + 1) : -static_cast<float>(i + 1);
  }

  void* upload_ptr = nullptr;
  upload_buffer->Map(0, nullptr, &upload_ptr);
  memcpy(upload_ptr, test_data.data(), buffer_size);
  upload_buffer->Unmap(0, nullptr);

  // Copy upload buffer to input buffer via D3D12
  command_allocator_->Reset();
  command_list_->Reset(command_allocator_.Get(), nullptr);
  command_list_->CopyBufferRegion(input_buffer.Get(), 0, upload_buffer.Get(), 0, buffer_size);
  command_list_->Close();

  ID3D12CommandList* cmd_lists[] = {command_list_.Get()};
  command_queue_->ExecuteCommandLists(1, cmd_lists);

  // Signal fence after D3D12 upload completes
  uint64_t upload_complete_value = 1;
  command_queue_->Signal(sync_fence.Get(), upload_complete_value);

  // Make CUDA wait for D3D12 upload to complete
  status = ort_interop_api_->WaitSemaphore(importer, sem_handle, ort_stream, upload_complete_value);
  ASSERT_EQ(status, nullptr) << "WaitSemaphore should succeed";

  // Setup ORT session with user_compute_stream
  Ort::SessionOptions session_options;
  session_options.SetExecutionMode(ORT_SEQUENTIAL);
  session_options.DisableMemPattern();
  session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

  // Add the NvTensorRtRtx EP with user stream
  status = ort_api_->SessionOptionsAppendExecutionProvider_V2(
      session_options, *ort_env, &ep_device_, 1,
      nullptr, nullptr, 0);
  ASSERT_EQ(status, nullptr);

  // Create session
  Ort::Session session(*ort_env, model_path.c_str(), session_options);

  // Create IoBinding and bind external tensors
  Ort::IoBinding io_binding(session);
  Ort::AllocatorWithDefaultOptions allocator;

  Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
  Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
  Ort::Value input(input_tensor);
  Ort::Value output(output_tensor);
  io_binding.BindInput(input_name.get(), input);
  io_binding.BindOutput(output_name.get(), output);
  io_binding.SynchronizeInputs();
  // Run inference. ORT submits all work to the stream before returning, so we signal the async semaphore below.
  Ort::RunOptions run_options;
  run_options.SetSyncStream(ort_stream);
  run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
  session.Run(run_options, io_binding);

  // Signal from CUDA that inference is complete
  uint64_t inference_complete_value = 2;
  status = ort_interop_api_->SignalSemaphore(importer, sem_handle, ort_stream, inference_complete_value);
  ASSERT_EQ(status, nullptr) << "SignalSemaphore should succeed";

  // Wait on D3D12 for CUDA inference to complete
  command_queue_->Wait(sync_fence.Get(), inference_complete_value);

  // Copy output to readback buffer
  ComPtr<ID3D12Resource> readback_buffer;
  D3D12ResourceHelper::CreateReadbackBuffer(d3d12_device_.Get(), buffer_size, &readback_buffer);

  command_allocator_->Reset();
  command_list_->Reset(command_allocator_.Get(), nullptr);
  command_list_->CopyBufferRegion(readback_buffer.Get(), 0, output_buffer.Get(), 0, buffer_size);
  command_list_->Close();

  command_queue_->ExecuteCommandLists(1, cmd_lists);
  D3D12ResourceHelper::FlushAndWait(d3d12_device_.Get(), command_queue_.Get());

  // Read back and verify ReLU output: max(0, x)
  std::vector<float> output_data(num_elements);
  void* readback_ptr = nullptr;
  readback_buffer->Map(0, nullptr, &readback_ptr);
  memcpy(output_data.data(), readback_ptr, buffer_size);
  readback_buffer->Unmap(0, nullptr);

  // Verify ReLU correctness
  for (size_t i = 0; i < num_elements; ++i) {
    float expected = std::max(0.0f, test_data[i]);
    EXPECT_FLOAT_EQ(output_data[i], expected)
        << "Mismatch at index " << i << ": input=" << test_data[i]
        << ", expected=" << expected << ", got=" << output_data[i];
  }

  // Note: io_binding takes ownership of input_tensor and output_tensor, so don't release them manually

  // Cleanup
  ort_interop_api_->ReleaseExternalSemaphoreHandle(sem_handle);
  ort_interop_api_->ReleaseExternalMemoryHandle(output_mem);
  ort_interop_api_->ReleaseExternalMemoryHandle(input_mem);
  CloseHandle(fence_handle);
  CloseHandle(output_handle);
  CloseHandle(input_handle);
  ort_api_->ReleaseSyncStream(ort_stream);
  ort_interop_api_->ReleaseExternalResourceImporter(importer);

  clearFileIfExists(model_path);
}

// Test: Full Inference with External Memory (E2E)
// This test validates the complete D3D12 to CUDA interop pipeline:
// 1. Create D3D12 shared resources and fences
// 2. Import them into CUDA via OrtExternalResourceImporter
// 3. Create ORT tensors from imported memory
// 4. Run inference with proper synchronization
// 5. Verify output correctness
TEST_F(NvExecutionProviderExternalResourceImporterTest, FullInferenceWithExternalMemoryCIG) {
  if (!IsD3D12Available()) {
    GTEST_SKIP() << "D3D12 not available";
  }

  // Push CIG context
  CUctxCigParam ctxCigParams;
  ctxCigParams.sharedDataType = CIG_DATA_TYPE_D3D12_COMMAND_QUEUE;
  ctxCigParams.sharedData = command_queue_.Get();
  CUctxCreateParams ctxParams = {nullptr, 0, &ctxCigParams};
  CUcontext cig_context;
  LUID d3d12_luid_struct = d3d12_device_->GetAdapterLuid();
  uint64_t d3d12_luid = (static_cast<uint64_t>(d3d12_luid_struct.HighPart) << 32) | d3d12_luid_struct.LowPart;
  int cuda_device_count = 0;
  int cuda_device_id = 0;
  bool found = false;
  ASSERT_EQ(cudaGetDeviceCount(&cuda_device_count), cudaSuccess);
  for (; cuda_device_id < cuda_device_count; ++cuda_device_id) {
    cudaDeviceProp prop;
    ASSERT_EQ(cudaGetDeviceProperties(&prop, cuda_device_id), cudaSuccess);
    uint64_t cuda_luid;
    std::memcpy(&cuda_luid, prop.luid, sizeof(char) * 8);
    if (cuda_luid == d3d12_luid) {
      found = true;
      break;
    }
  }
  ASSERT_TRUE(found);
  CudaDriverLoader cuda_driver_loader;
  ASSERT_TRUE(cuda_driver_loader.IsLoaded());
  ASSERT_EQ(cuda_driver_loader.cuCtxCreate_v4_fn(&cig_context, &ctxParams, 0, cuda_device_id), CUDA_SUCCESS);
  ASSERT_EQ(cuda_driver_loader.cuCtxSetCurrent_fn(cig_context), CUDA_SUCCESS);

  if (!IsEPAvailable()) {
    GTEST_SKIP() << "NvTensorRtRtx EP not available";
  }
  {
    // Create a simple ReLU model using shared utility pattern
    PathString model_path = ORT_TSTR("external_mem_relu_test.onnx");
    {
      onnxruntime::Model model("relu_test", false, DefaultLoggingManager().DefaultLogger());
      auto& graph = model.MainGraph();

      ONNX_NAMESPACE::TypeProto tensor_type;
      tensor_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
      tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
      tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(64);
      tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(64);

      auto& input_arg = graph.GetOrCreateNodeArg("X", &tensor_type);
      auto& output_arg = graph.GetOrCreateNodeArg("Y", &tensor_type);
      graph.AddNode("relu", "Relu", "ReLU operation", {&input_arg}, {&output_arg});

      ASSERT_STATUS_OK(graph.Resolve());
      ASSERT_STATUS_OK(onnxruntime::Model::Save(model, model_path));
    }

    const int64_t batch = 1, channels = 3, dim = 64;
    const int64_t shape[] = {batch, channels, dim, dim};
    const size_t num_elements = batch * channels * dim * dim;
    const size_t buffer_size = num_elements * sizeof(float);

    // Create external resource importer
    OrtExternalResourceImporter* importer = nullptr;
    OrtStatus* status = ort_interop_api_->CreateExternalResourceImporterForDevice(ep_device_, &importer);
    if (status != nullptr || importer == nullptr) {
      if (status != nullptr) ort_api_->ReleaseStatus(status);
      clearFileIfExists(model_path);
      GTEST_SKIP() << "External resource import not supported";
    }

    // Create CUDA stream via ORT
    OrtSyncStream* ort_stream = nullptr;
    status = ort_api_->CreateSyncStreamForEpDevice(ep_device_, nullptr, &ort_stream);
    ASSERT_EQ(status, nullptr);

    // Create shared D3D12 buffers for input and output
    ComPtr<ID3D12Resource> input_buffer, output_buffer;
    D3D12ResourceHelper::CreateSharedBuffer(d3d12_device_.Get(), buffer_size, &input_buffer);
    D3D12ResourceHelper::CreateSharedBuffer(d3d12_device_.Get(), buffer_size, &output_buffer);

    // Create shared handles for cross-API import
    HANDLE input_handle = nullptr, output_handle = nullptr;
    d3d12_device_->CreateSharedHandle(input_buffer.Get(), nullptr, GENERIC_ALL, nullptr, &input_handle);
    d3d12_device_->CreateSharedHandle(output_buffer.Get(), nullptr, GENERIC_ALL, nullptr, &output_handle);

    // Import memory into CUDA
    OrtExternalMemoryDescriptor input_mem_desc = {};
    input_mem_desc.version = ORT_API_VERSION;
    input_mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    input_mem_desc.native_handle = input_handle;
    input_mem_desc.size_bytes = buffer_size;
    input_mem_desc.offset_bytes = 0;

    OrtExternalMemoryDescriptor output_mem_desc = input_mem_desc;
    output_mem_desc.native_handle = output_handle;

    OrtExternalMemoryHandle *input_mem = nullptr, *output_mem = nullptr;
    status = ort_interop_api_->ImportMemory(importer, &input_mem_desc, &input_mem);
    ASSERT_EQ(status, nullptr) << "ImportMemory for input should succeed (proves cuImportExternalMemory called)";
    status = ort_interop_api_->ImportMemory(importer, &output_mem_desc, &output_mem);
    ASSERT_EQ(status, nullptr) << "ImportMemory for output should succeed";

    // Create ORT tensors from imported memory
    OrtExternalTensorDescriptor tensor_desc = {};
    tensor_desc.version = ORT_API_VERSION;
    tensor_desc.element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    tensor_desc.shape = shape;
    tensor_desc.rank = 4;
    tensor_desc.offset_bytes = 0;

    OrtValue *input_tensor = nullptr, *output_tensor = nullptr;
    status = ort_interop_api_->CreateTensorFromMemory(importer, input_mem, &tensor_desc, &input_tensor);
    ASSERT_EQ(status, nullptr);
    status = ort_interop_api_->CreateTensorFromMemory(importer, output_mem, &tensor_desc, &output_tensor);
    ASSERT_EQ(status, nullptr);

    // Verify the tensor data pointers are CUDA device memory
    void* input_data_ptr = nullptr;
    void* output_data_ptr = nullptr;
    status = ort_api_->GetTensorMutableData(input_tensor, &input_data_ptr);
    ASSERT_EQ(status, nullptr);
    status = ort_api_->GetTensorMutableData(output_tensor, &output_data_ptr);
    ASSERT_EQ(status, nullptr);

    cudaPointerAttributes input_attrs, output_attrs;
    ASSERT_EQ(cudaPointerGetAttributes(&input_attrs, input_data_ptr), cudaSuccess);
    ASSERT_EQ(cudaPointerGetAttributes(&output_attrs, output_data_ptr), cudaSuccess);
    EXPECT_EQ(input_attrs.type, cudaMemoryTypeDevice) << "Input tensor must be CUDA device memory";
    EXPECT_EQ(output_attrs.type, cudaMemoryTypeDevice) << "Output tensor must be CUDA device memory";

    // Create D3D12 fence for bidirectional synchronization
    ComPtr<ID3D12Fence> sync_fence;
    d3d12_device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&sync_fence));
    HANDLE fence_handle = nullptr;
    d3d12_device_->CreateSharedHandle(sync_fence.Get(), nullptr, GENERIC_ALL, nullptr, &fence_handle);

    OrtExternalSemaphoreDescriptor sem_desc = {};
    sem_desc.version = ORT_API_VERSION;
    sem_desc.type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
    sem_desc.native_handle = fence_handle;

    OrtExternalSemaphoreHandle* sem_handle = nullptr;
    status = ort_interop_api_->ImportSemaphore(importer, &sem_desc, &sem_handle);
    ASSERT_EQ(status, nullptr) << "ImportSemaphore should succeed";

    // Setup test data via D3D12 upload buffer
    ComPtr<ID3D12Resource> upload_buffer;
    D3D12ResourceHelper::CreateUploadBuffer(d3d12_device_.Get(), buffer_size, &upload_buffer);

    // Generate test data: alternating positive and negative values for ReLU verification
    std::vector<float> test_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
      test_data[i] = (i % 2 == 0) ? static_cast<float>(i + 1) : -static_cast<float>(i + 1);
    }

    void* upload_ptr = nullptr;
    upload_buffer->Map(0, nullptr, &upload_ptr);
    memcpy(upload_ptr, test_data.data(), buffer_size);
    upload_buffer->Unmap(0, nullptr);

    // Copy upload buffer to input buffer via D3D12
    command_allocator_->Reset();
    command_list_->Reset(command_allocator_.Get(), nullptr);
    command_list_->CopyBufferRegion(input_buffer.Get(), 0, upload_buffer.Get(), 0, buffer_size);
    command_list_->Close();

    ID3D12CommandList* cmd_lists[] = {command_list_.Get()};
    command_queue_->ExecuteCommandLists(1, cmd_lists);

    // Signal fence after D3D12 upload completes
    uint64_t upload_complete_value = 1;
    command_queue_->Signal(sync_fence.Get(), upload_complete_value);

    // Make CUDA wait for D3D12 upload to complete
    status = ort_interop_api_->WaitSemaphore(importer, sem_handle, ort_stream, upload_complete_value);
    ASSERT_EQ(status, nullptr) << "WaitSemaphore should succeed";

    // Setup ORT session with user_compute_stream
    Ort::SessionOptions session_options;
    session_options.SetExecutionMode(ORT_SEQUENTIAL);
    session_options.DisableMemPattern();
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    // Configure to use our CUDA stream
    char stream_address[32];
    size_t stream_addr_val = reinterpret_cast<size_t>(ort_api_->SyncStream_GetHandle(ort_stream));
    sprintf(stream_address, "%llu", static_cast<uint64_t>(stream_addr_val));
    const char* option_keys[] = {
        // TODO we should no longer require to set the compute stream at this point but there are too many cudaSetDevice calls from allocators and stream handling (NVBUG 5822116)
        onnxruntime::nv::provider_option_names::kUserComputeStream,
        onnxruntime::nv::provider_option_names::kHasUserComputeStream,
        onnxruntime::nv::provider_option_names::kMaxSharedMemSize,
        // TRT will create itss own context to create streams if we do not manually provide aux streams
        onnxruntime::nv::provider_option_names::kLengthAuxStreamArray,
        onnxruntime::nv::provider_option_names::kUserAuxStreamArray,
        onnxruntime::nv::provider_option_names::kCudaGraphEnable,
    };
    char aux_stream_address[32];
    size_t aux_streams[] = {stream_addr_val};
    sprintf(aux_stream_address, "%llu", reinterpret_cast<uint64_t>(aux_streams));
    std::string max_shared_mem_size = std::to_string(1024 * 28);  // 28 KiB
    const char* option_values[] = {
        stream_address,
        "1",
        max_shared_mem_size.c_str(),
        "1",
        aux_stream_address,
        "0"};

    // Add the NvTensorRtRtx EP with user stream
    status = ort_api_->SessionOptionsAppendExecutionProvider_V2(
        session_options, *ort_env, &ep_device_, 1, option_keys, option_values, 6);
    ASSERT_EQ(status, nullptr);

    // Create session
    Ort::Session session(*ort_env, model_path.c_str(), session_options);

    // Create IoBinding and bind external tensors
    Ort::IoBinding io_binding(session);
    Ort::AllocatorWithDefaultOptions allocator;

    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);

    Ort::Value input(input_tensor);
    Ort::Value output(output_tensor);
    io_binding.BindInput(input_name.get(), input);
    io_binding.BindOutput(output_name.get(), output);
    io_binding.SynchronizeInputs();
    // Run inference. ORT submits all work to the stream before returning, so we signal the async semaphore below.
    Ort::RunOptions run_options;
    run_options.SetSyncStream(ort_stream);
    run_options.AddConfigEntry(kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
    session.Run(run_options, io_binding);

    // Signal from CUDA that inference is complete
    uint64_t inference_complete_value = 2;
    status = ort_interop_api_->SignalSemaphore(importer, sem_handle, ort_stream, inference_complete_value);
    ASSERT_EQ(status, nullptr) << "SignalSemaphore should succeed";

    // Wait on D3D12 for CUDA inference to complete
    command_queue_->Wait(sync_fence.Get(), inference_complete_value);

    // Copy output to readback buffer
    ComPtr<ID3D12Resource> readback_buffer;
    D3D12ResourceHelper::CreateReadbackBuffer(d3d12_device_.Get(), buffer_size, &readback_buffer);

    command_allocator_->Reset();
    command_list_->Reset(command_allocator_.Get(), nullptr);
    command_list_->CopyBufferRegion(readback_buffer.Get(), 0, output_buffer.Get(), 0, buffer_size);
    command_list_->Close();

    command_queue_->ExecuteCommandLists(1, cmd_lists);
    D3D12ResourceHelper::FlushAndWait(d3d12_device_.Get(), command_queue_.Get());

    // Read back and verify ReLU output: max(0, x)
    std::vector<float> output_data(num_elements);
    void* readback_ptr = nullptr;
    readback_buffer->Map(0, nullptr, &readback_ptr);
    memcpy(output_data.data(), readback_ptr, buffer_size);
    readback_buffer->Unmap(0, nullptr);

    // Verify ReLU correctness
    for (size_t i = 0; i < num_elements; ++i) {
      float expected = std::max(0.0f, test_data[i]);
      EXPECT_FLOAT_EQ(output_data[i], expected)
          << "Mismatch at index " << i << ": input=" << test_data[i]
          << ", expected=" << expected << ", got=" << output_data[i];
    }
    CUcontext cu_context;
    ASSERT_EQ(cuda_driver_loader.cuCtxGetCurrent_fn(&cu_context), CUDA_SUCCESS);
    ASSERT_EQ(cu_context, cig_context);
    // Note: io_binding takes ownership of input_tensor and output_tensor, so don't release them manually

    // Cleanup
    ort_interop_api_->ReleaseExternalSemaphoreHandle(sem_handle);
    ort_interop_api_->ReleaseExternalMemoryHandle(output_mem);
    ort_interop_api_->ReleaseExternalMemoryHandle(input_mem);
    CloseHandle(fence_handle);
    CloseHandle(output_handle);
    CloseHandle(input_handle);
    ort_api_->ReleaseSyncStream(ort_stream);
    ort_interop_api_->ReleaseExternalResourceImporter(importer);
    clearFileIfExists(model_path);
  }
  // all associated objects with the context must be destroyed before destroying the CIG context which ORT inherited
  ASSERT_EQ(cuda_driver_loader.cuCtxDestroy_fn(cig_context), CUDA_SUCCESS);
}

#endif  // _WIN32

}  // namespace test
}  // namespace onnxruntime
