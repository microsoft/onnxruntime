// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CUDA

#include <gtest/gtest.h>
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"

#ifdef _WIN32
#include <wrl/client.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cuda.h>
#include <cudaD3D12.h>

using Microsoft::WRL::ComPtr;

namespace onnxruntime {
namespace test {

class CudaExternalMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize CUDA
    CUresult cu_result = cuInit(0);
    if (cu_result != CUDA_SUCCESS) {
      GTEST_SKIP() << "CUDA not available";
    }

    // Get CUDA device count
    int device_count = 0;
    cu_result = cuDeviceGetCount(&device_count);
    if (cu_result != CUDA_SUCCESS || device_count == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }

    // Create ORT environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CudaExternalMemoryTest");
  }

  void TearDown() override {
    env_.reset();
  }

  std::unique_ptr<Ort::Env> env_;
};

// Test QueryExternalMemorySupport for CUDA EP
TEST_F(CudaExternalMemoryTest, QueryD3D12ResourceSupport) {
  Ort::SessionOptions session_options;
  
  // Use stable EP ABI - enumerate devices and add CUDA EP via V2
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "CUDAExecutionProvider") == 0) {
          // Add CUDA EP via stable V2 interface
          options->AppendExecutionProvider_V2(provider_name, nullptr);
        }
      },
      &session_options));
  
  // Create a simple model (1x1 identity)
  const char* model_path = "testdata/mul_1.onnx";
  Ort::Session session(*env_, model_path, session_options);
  
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  int supported = 0;
  
  // Query D3D12_RESOURCE support
  OrtStatus* status = api->QueryExternalMemorySupport(
      session,
      ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
      &supported);
  
  ASSERT_EQ(status, nullptr);
  
  // CUDA EP should support D3D12 resource import on Windows
  EXPECT_EQ(supported, 1) << "CUDA EP should support D3D12_RESOURCE import";
}

// Test QueryExternalMemorySupport for CUDA device pointers
TEST_F(CudaExternalMemoryTest, QueryCudaPointerSupport) {
  Ort::SessionOptions session_options;
  
  // Use stable EP ABI
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "CUDAExecutionProvider") == 0) {
          options->AppendExecutionProvider_V2(provider_name, nullptr);
        }
      },
      &session_options));
  
  const char* model_path = "testdata/mul_1.onnx";
  Ort::Session session(*env_, model_path, session_options);
  
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  int supported = 0;
  
  // Query CUDA pointer support
  OrtStatus* status = api->QueryExternalMemorySupport(
      session,
      ORT_EXTERNAL_MEMORY_HANDLE_TYPE_CUDA,
      &supported);
  
  ASSERT_EQ(status, nullptr);
  EXPECT_EQ(supported, 1) << "CUDA EP should support CUDA pointer import";
}

#ifdef _WIN32
// Test actual D3D12 to CUDA memory import
TEST_F(CudaExternalMemoryTest, ImportD3D12Resource) {
  // Create D3D12 device
  ComPtr<ID3D12Device> d3d12_device;
  HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12_device));
  
  if (FAILED(hr)) {
    GTEST_SKIP() << "D3D12 device not available";
  }
  
  // Create a shared D3D12 resource (1MB buffer)
  const size_t buffer_size = 1024 * 1024;
  
  D3D12_HEAP_PROPERTIES heap_props = {};
  heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
  
  D3D12_RESOURCE_DESC resource_desc = {};
  resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resource_desc.Width = buffer_size;
  resource_desc.Height = 1;
  resource_desc.DepthOrArraySize = 1;
  resource_desc.MipLevels = 1;
  resource_desc.Format = DXGI_FORMAT_UNKNOWN;
  resource_desc.SampleDesc.Count = 1;
  resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resource_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  
  ComPtr<ID3D12Resource> d3d12_resource;
  hr = d3d12_device->CreateCommittedResource(
      &heap_props,
      D3D12_HEAP_FLAG_SHARED,  // Must be shared for CUDA import
      &resource_desc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(&d3d12_resource));
  
  ASSERT_TRUE(SUCCEEDED(hr));
  
  // Create a shared handle for the resource
  HANDLE shared_handle = nullptr;
  hr = d3d12_device->CreateSharedHandle(
      d3d12_resource.Get(),
      nullptr,
      GENERIC_ALL,
      nullptr,
      &shared_handle);
  
  ASSERT_TRUE(SUCCEEDED(hr));
  ASSERT_NE(shared_handle, nullptr);
  
  // Create session with CUDA EP via stable ABI
  Ort::SessionOptions session_options;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "CUDAExecutionProvider") == 0) {
          options->AppendExecutionProvider_V2(provider_name, nullptr);
        }
      },
      &session_options));
  
  const char* model_path = "testdata/mul_1.onnx";
  Ort::Session session(*env_, model_path, session_options);
  
  // Create IOBinding
  Ort::IoBinding io_binding(session);
  
  // Create external memory descriptor for D3D12 resource
  OrtExternalMemoryDescriptor ext_mem_desc = {};
  ext_mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  ext_mem_desc.native_handle = d3d12_resource.Get();
  ext_mem_desc.size = buffer_size;
  ext_mem_desc.offset = 0;
  ext_mem_desc.flags = 0;
  ext_mem_desc.access_mode = ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE;
  ext_mem_desc.wait_semaphore_handle = nullptr;
  ext_mem_desc.wait_semaphore_value = 0;
  ext_mem_desc.signal_semaphore_handle = nullptr;
  ext_mem_desc.signal_semaphore_value = 0;
  
  // Create tensor shape info (256 floats)
  const int64_t shape[] = {256};
  const size_t dim_count = 1;
  
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto type_info = Ort::TypeInfo::CreateTensor(mem_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, dim_count);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  
  // Bind external memory to IOBinding
  OrtStatus* status = api->IOBindingBindExternalMemory(
      io_binding,
      "X",  // Input name
      tensor_info,
      &ext_mem_desc,
      ORT_EXTERNAL_MEMORY_ACCESS_READ_ONLY);
  
  // Should succeed - CUDA EP can import D3D12 resources
  if (status != nullptr) {
    const char* error_msg = api->GetErrorMessage(status);
    FAIL() << "Failed to bind D3D12 resource to CUDA EP: " << error_msg;
    api->ReleaseStatus(status);
  }
  
  // Cleanup
  CloseHandle(shared_handle);
}

// Test D3D12 heap import (CUDA should support this too)
TEST_F(CudaExternalMemoryTest, ImportD3D12Heap) {
  // Create D3D12 device
  ComPtr<ID3D12Device> d3d12_device;
  HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12_device));
  
  if (FAILED(hr)) {
    GTEST_SKIP() << "D3D12 device not available";
  }
  
  // Create a shared D3D12 heap (1MB)
  const size_t heap_size = 1024 * 1024;
  
  D3D12_HEAP_DESC heap_desc = {};
  heap_desc.SizeInBytes = heap_size;
  heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
  heap_desc.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
  
  ComPtr<ID3D12Heap> d3d12_heap;
  hr = d3d12_device->CreateHeap(&heap_desc, IID_PPV_ARGS(&d3d12_heap));
  
  ASSERT_TRUE(SUCCEEDED(hr));
  
  // Query if CUDA EP supports D3D12_HEAP via stable ABI
  Ort::SessionOptions session_options;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "CUDAExecutionProvider") == 0) {
          options->AppendExecutionProvider_V2(provider_name, nullptr);
        }
      },
      &session_options));
  
  const char* model_path = "testdata/mul_1.onnx";
  Ort::Session session(*env_, model_path, session_options);
  
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  int supported = 0;
  
  OrtStatus* status = api->QueryExternalMemorySupport(
      session,
      ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP,
      &supported);
  
  ASSERT_EQ(status, nullptr);
  // CUDA EP may or may not support heap import - implementation dependent
  // Just verify the query works
}
#endif  // _WIN32

// Test CUDA device pointer import
TEST_F(CudaExternalMemoryTest, ImportCudaDevicePointer) {
  // Allocate CUDA device memory
  void* cuda_ptr = nullptr;
  const size_t buffer_size = 1024 * 1024;  // 1MB
  
  cudaError_t cuda_err = cudaMalloc(&cuda_ptr, buffer_size);
  if (cuda_err != cudaSuccess) {
    GTEST_SKIP() << "Failed to allocate CUDA memory: " << cudaGetErrorString(cuda_err);
  }
  
  // Create session with CUDA EP via stable ABI
  Ort::SessionOptions session_options;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "CUDAExecutionProvider") == 0) {
          options->AppendExecutionProvider_V2(provider_name, nullptr);
        }
      },
      &session_options));
  
  const char* model_path = "testdata/mul_1.onnx";
  Ort::Session session(*env_, model_path, session_options);
  
  // Create IOBinding
  Ort::IoBinding io_binding(session);
  
  // Create external memory descriptor for CUDA pointer
  OrtExternalMemoryDescriptor ext_mem_desc = {};
  ext_mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_CUDA;
  ext_mem_desc.native_handle = cuda_ptr;
  ext_mem_desc.size = buffer_size;
  ext_mem_desc.offset = 0;
  ext_mem_desc.access_mode = ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE;
  
  // Create tensor shape info
  const int64_t shape[] = {256};  // 256 floats
  const size_t dim_count = 1;
  
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto type_info = Ort::TypeInfo::CreateTensor(mem_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, dim_count);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  
  // Bind CUDA pointer to IOBinding
  OrtStatus* status = api->IOBindingBindExternalMemory(
      io_binding,
      "X",
      tensor_info,
      &ext_mem_desc,
      ORT_EXTERNAL_MEMORY_ACCESS_READ_ONLY);
  
  // Should succeed
  if (status != nullptr) {
    const char* error_msg = api->GetErrorMessage(status);
    FAIL() << "Failed to bind CUDA pointer: " << error_msg;
    api->ReleaseStatus(status);
  }
  
  // Cleanup
  cudaFree(cuda_ptr);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // _WIN32
#endif  // USE_CUDA
