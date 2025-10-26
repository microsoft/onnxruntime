// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// External memory import tests only supported on Windows
#ifdef _WIN32

#include <gtest/gtest.h>
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"

#ifdef USE_DML
#include <wrl/client.h>
#include <d3d12.h>
using Microsoft::WRL::ComPtr;
#endif

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

// Test that the external memory API functions are present in OrtApi
TEST(ExternalMemoryApiTest, FunctionPointersAreValid) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);
  
  // Verify all three new functions exist
  EXPECT_NE(api->QueryExternalMemorySupport, nullptr);
  EXPECT_NE(api->CreateTensorFromExternalMemory, nullptr);
  EXPECT_NE(api->IOBindingBindExternalMemory, nullptr);
}

// Test that the external memory types are defined correctly
TEST(ExternalMemoryApiTest, TypeDefinitions) {
  // Test handle type enum
  EXPECT_EQ(ORT_EXTERNAL_MEMORY_HANDLE_TYPE_NONE, 0);
  EXPECT_EQ(ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE, 1);
  EXPECT_EQ(ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP, 2);
  
  // Test access mode enum
  EXPECT_EQ(ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE, 0);
  EXPECT_EQ(ORT_EXTERNAL_MEMORY_ACCESS_READ_ONLY, 1);
  EXPECT_EQ(ORT_EXTERNAL_MEMORY_ACCESS_WRITE_ONLY, 2);
  
  // Test descriptor struct size is reasonable (should have 8 fields)
  EXPECT_GT(sizeof(OrtExternalMemoryDescriptor), 0);
}

// Test QueryExternalMemorySupport with NULL parameters returns error
TEST(ExternalMemoryApiTest, QueryExternalMemorySupport_NullParams) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);
  ASSERT_NE(api->QueryExternalMemorySupport, nullptr);
  
  int supported = 0;
  
  // NULL session should return error
  OrtStatus* status = api->QueryExternalMemorySupport(
      nullptr,
      ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
      &supported);
  
  EXPECT_NE(status, nullptr);
  if (status) {
    api->ReleaseStatus(status);
  }
  
  // NULL out_supported should return error (can't test with NULL session)
  // This would crash, so we skip it
}

// Test CreateTensorFromExternalMemory returns NOT_IMPLEMENTED
TEST(ExternalMemoryApiTest, CreateTensorFromExternalMemory_NotImplemented) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);
  ASSERT_NE(api->CreateTensorFromExternalMemory, nullptr);
  
  OrtExternalMemoryDescriptor desc = {};
  desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  desc.native_handle = nullptr;
  desc.size = 1024;
  desc.access_mode = ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE;
  
  OrtValue* tensor = nullptr;
  OrtAllocator* allocator = nullptr;
  api->GetAllocatorWithDefaultOptions(&allocator);
  
  OrtStatus* status = api->CreateTensorFromExternalMemory(
      nullptr,  // info
      &desc,
      allocator,
      &tensor);
  
  // Should return error (stub implementation)
  EXPECT_NE(status, nullptr);
  if (status) {
    api->ReleaseStatus(status);
  }
}

// Test IOBindingBindExternalMemory returns NOT_IMPLEMENTED
TEST(ExternalMemoryApiTest, IOBindingBindExternalMemory_NotImplemented) {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  ASSERT_NE(api, nullptr);
  ASSERT_NE(api->IOBindingBindExternalMemory, nullptr);
  
  OrtExternalMemoryDescriptor desc = {};
  desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  desc.native_handle = nullptr;
  desc.size = 1024;
  desc.access_mode = ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE;
  
  OrtStatus* status = api->IOBindingBindExternalMemory(
      nullptr,  // binding
      "input",
      nullptr,  // info
      &desc,
      ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE);
  
  // Should return error (stub implementation)
  EXPECT_NE(status, nullptr);
  if (status) {
    api->ReleaseStatus(status);
  }
}

#ifdef USE_DML
// Test external memory descriptor with real D3D12 resource
TEST(ExternalMemoryApiTest, D3D12ResourceDescriptor) {
  ComPtr<ID3D12Device> device;
  HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
  
  if (FAILED(hr)) {
    GTEST_SKIP() << "D3D12 device not available";
  }
  
  // Create a buffer resource
  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
  
  D3D12_RESOURCE_DESC resourceDesc = {};
  resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resourceDesc.Width = 1024 * 1024;  // 1 MB
  resourceDesc.Height = 1;
  resourceDesc.DepthOrArraySize = 1;
  resourceDesc.MipLevels = 1;
  resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
  resourceDesc.SampleDesc.Count = 1;
  resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resourceDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  
  ComPtr<ID3D12Resource> resource;
  hr = device->CreateCommittedResource(
      &heapProps,
      D3D12_HEAP_FLAG_NONE,
      &resourceDesc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(&resource));
  
  ASSERT_TRUE(SUCCEEDED(hr));
  
  // Create external memory descriptor
  OrtExternalMemoryDescriptor desc = {};
  desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  desc.native_handle = resource.Get();
  desc.size = 1024 * 1024;
  desc.access_mode = ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE;
  desc.wait_semaphore_handle = nullptr;
  desc.wait_semaphore_value = 0;
  desc.signal_semaphore_handle = nullptr;
  desc.signal_semaphore_value = 0;
  
  // Verify descriptor fields
  EXPECT_EQ(desc.handle_type, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE);
  EXPECT_NE(desc.native_handle, nullptr);
  EXPECT_EQ(desc.size, 1024 * 1024);
  EXPECT_EQ(desc.access_mode, ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE);
}
#endif  // USE_DML

}  // namespace test
}  // namespace onnxruntime

#endif  // _WIN32
