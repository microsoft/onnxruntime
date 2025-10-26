// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_NV_EXECUTION_PROVIDER

#include <gtest/gtest.h>
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"

#ifdef _WIN32
#include <wrl/client.h>
#include <d3d12.h>
#include <dxgi1_6.h>

using Microsoft::WRL::ComPtr;
#endif

namespace onnxruntime {
namespace test {

class NvExecutionProviderExternalMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create ORT environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NvExecutionProviderExternalMemoryTest");
  }

  void TearDown() override {
    env_.reset();
  }

  std::unique_ptr<Ort::Env> env_;
};

// Test QueryExternalMemorySupport for D3D12 resources
TEST_F(NvExecutionProviderExternalMemoryTest, QueryD3D12ResourceSupport) {
  Ort::SessionOptions session_options;
  
  // Use stable EP ABI - enumerate and add NV EP via V2
  bool nv_ep_found = false;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* found = static_cast<bool*>(user_data);
        if (strcmp(provider_name, "NvExecutionProvider") == 0) {
          *found = true;
        }
      },
      &nv_ep_found));
  
  if (!nv_ep_found) {
    GTEST_SKIP() << "NV Execution Provider not available";
  }
  
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "NvExecutionProvider") == 0) {
          options->AppendExecutionProvider_V2(provider_name, nullptr);
        }
      },
      &session_options));
  
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
  
  // NV EP should support D3D12 resource import on Windows
  EXPECT_EQ(supported, 1) << "NV EP should support D3D12_RESOURCE import";
}

// Test QueryExternalMemorySupport for D3D12 heaps
TEST_F(NvExecutionProviderExternalMemoryTest, QueryD3D12HeapSupport) {
  Ort::SessionOptions session_options;
  
  bool nv_ep_found = false;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* found = static_cast<bool*>(user_data);
        if (strcmp(provider_name, "NvExecutionProvider") == 0) {
          *found = true;
        }
      },
      &nv_ep_found));
  
  if (!nv_ep_found) {
    GTEST_SKIP() << "NV Execution Provider not available";
  }
  
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "NvExecutionProvider") == 0) {
          options->AppendExecutionProvider_V2(provider_name, nullptr);
        }
      },
      &session_options));
  
  const char* model_path = "testdata/mul_1.onnx";
  Ort::Session session(*env_, model_path, session_options);
  
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  int supported = 0;
  
  // Query D3D12_HEAP support
  OrtStatus* status = api->QueryExternalMemorySupport(
      session,
      ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP,
      &supported);
  
  ASSERT_EQ(status, nullptr);
  // NV EP may or may not support heap import - implementation dependent
}

// Test QueryExternalMemorySupport for CUDA pointers
TEST_F(NvExecutionProviderExternalMemoryTest, QueryCudaPointerSupport) {
  Ort::SessionOptions session_options;
  
  bool nv_ep_found = false;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* found = static_cast<bool*>(user_data);
        if (strcmp(provider_name, "NvExecutionProvider") == 0) {
          *found = true;
        }
      },
      &nv_ep_found));
  
  if (!nv_ep_found) {
    GTEST_SKIP() << "NV Execution Provider not available";
  }
  
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "NvExecutionProvider") == 0) {
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
  // NV EP likely supports CUDA pointers if it wraps CUDA
  EXPECT_EQ(supported, 1) << "NV EP should support CUDA pointer import";
}

#ifdef _WIN32
// Test actual D3D12 resource import with NV EP
TEST_F(NvExecutionProviderExternalMemoryTest, ImportD3D12Resource) {
  // Check if NV EP is available
  bool nv_ep_found = false;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* found = static_cast<bool*>(user_data);
        if (strcmp(provider_name, "NvExecutionProvider") == 0) {
          *found = true;
        }
      },
      &nv_ep_found));
  
  if (!nv_ep_found) {
    GTEST_SKIP() << "NV Execution Provider not available";
  }
  
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
      D3D12_HEAP_FLAG_SHARED,
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
  
  // Create session with NV EP via stable ABI
  Ort::SessionOptions session_options;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "NvExecutionProvider") == 0) {
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
  
  // Should succeed - NV EP can import D3D12 resources
  if (status != nullptr) {
    const char* error_msg = api->GetErrorMessage(status);
    FAIL() << "Failed to bind D3D12 resource to NV EP: " << error_msg;
    api->ReleaseStatus(status);
  }
  
  // Cleanup
  CloseHandle(shared_handle);
}

// Test D3D12 resource with different access modes
TEST_F(NvExecutionProviderExternalMemoryTest, D3D12ResourceAccessModes) {
  // Check if NV EP is available
  bool nv_ep_found = false;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* found = static_cast<bool*>(user_data);
        if (strcmp(provider_name, "NvExecutionProvider") == 0) {
          *found = true;
        }
      },
      &nv_ep_found));
  
  if (!nv_ep_found) {
    GTEST_SKIP() << "NV Execution Provider not available";
  }
  
  // Create D3D12 device
  ComPtr<ID3D12Device> d3d12_device;
  HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12_device));
  if (FAILED(hr)) {
    GTEST_SKIP() << "D3D12 device not available";
  }
  
  // Create a shared D3D12 resource
  const size_t buffer_size = 1024 * sizeof(float);
  
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
      D3D12_HEAP_FLAG_SHARED,
      &resource_desc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(&d3d12_resource));
  
  ASSERT_TRUE(SUCCEEDED(hr));
  
  // Create session with NV EP
  Ort::SessionOptions session_options;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "NvExecutionProvider") == 0) {
          options->AppendExecutionProvider_V2(provider_name, nullptr);
        }
      },
      &session_options));
  
  const char* model_path = "testdata/mul_1.onnx";
  Ort::Session session(*env_, model_path, session_options);
  Ort::IoBinding io_binding(session);
  
  const int64_t shape[] = {256};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto type_info = Ort::TypeInfo::CreateTensor(mem_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, 1);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  
  // Test READ_ONLY access
  {
    OrtExternalMemoryDescriptor desc = {};
    desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    desc.native_handle = d3d12_resource.Get();
    desc.size = buffer_size;
    desc.access_mode = ORT_EXTERNAL_MEMORY_ACCESS_READ_ONLY;
    
    OrtStatus* status = api->IOBindingBindExternalMemory(
        io_binding, "X", tensor_info, &desc, ORT_EXTERNAL_MEMORY_ACCESS_READ_ONLY);
    
    EXPECT_EQ(status, nullptr);
    if (status) api->ReleaseStatus(status);
  }
  
  // Test WRITE_ONLY access
  {
    OrtExternalMemoryDescriptor desc = {};
    desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    desc.native_handle = d3d12_resource.Get();
    desc.size = buffer_size;
    desc.access_mode = ORT_EXTERNAL_MEMORY_ACCESS_WRITE_ONLY;
    
    OrtStatus* status = api->IOBindingBindExternalMemory(
        io_binding, "Y", tensor_info, &desc, ORT_EXTERNAL_MEMORY_ACCESS_WRITE_ONLY);
    
    EXPECT_EQ(status, nullptr);
    if (status) api->ReleaseStatus(status);
  }
  
  // Test READ_WRITE access
  {
    OrtExternalMemoryDescriptor desc = {};
    desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    desc.native_handle = d3d12_resource.Get();
    desc.size = buffer_size;
    desc.access_mode = ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE;
    
    OrtStatus* status = api->IOBindingBindExternalMemory(
        io_binding, "Z", tensor_info, &desc, ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE);
    
    EXPECT_EQ(status, nullptr);
    if (status) api->ReleaseStatus(status);
  }
}
#endif  // _WIN32

// Test error cases
TEST_F(NvExecutionProviderExternalMemoryTest, ErrorCases) {
  // Check if NV EP is available
  bool nv_ep_found = false;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* found = static_cast<bool*>(user_data);
        if (strcmp(provider_name, "NvExecutionProvider") == 0) {
          *found = true;
        }
      },
      &nv_ep_found));
  
  if (!nv_ep_found) {
    GTEST_SKIP() << "NV Execution Provider not available";
  }
  
  Ort::SessionOptions session_options;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "NvExecutionProvider") == 0) {
          options->AppendExecutionProvider_V2(provider_name, nullptr);
        }
      },
      &session_options));
  
  const char* model_path = "testdata/mul_1.onnx";
  Ort::Session session(*env_, model_path, session_options);
  Ort::IoBinding io_binding(session);
  
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  
  // Test NULL descriptor
  {
    const int64_t shape[] = {256};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto type_info = Ort::TypeInfo::CreateTensor(mem_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, 1);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    
    OrtStatus* status = api->IOBindingBindExternalMemory(
        io_binding, "X", tensor_info, nullptr, ORT_EXTERNAL_MEMORY_ACCESS_READ_ONLY);
    
    EXPECT_NE(status, nullptr) << "Should fail with NULL descriptor";
    if (status) api->ReleaseStatus(status);
  }
  
  // Test unsupported handle type (HIP on Windows)
  {
    OrtExternalMemoryDescriptor desc = {};
    desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HIP;  // Not supported on Windows NV EP
    desc.native_handle = reinterpret_cast<void*>(0x1000);
    desc.size = 1024;
    
    const int64_t shape[] = {256};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto type_info = Ort::TypeInfo::CreateTensor(mem_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, 1);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    
    OrtStatus* status = api->IOBindingBindExternalMemory(
        io_binding, "X", tensor_info, &desc, ORT_EXTERNAL_MEMORY_ACCESS_READ_ONLY);
    
    EXPECT_NE(status, nullptr) << "Should fail with unsupported handle type";
    if (status) api->ReleaseStatus(status);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_NV_EXECUTION_PROVIDER
