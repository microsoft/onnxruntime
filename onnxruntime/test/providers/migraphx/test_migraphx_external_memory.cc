// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_MIGRAPHX

#include <gtest/gtest.h>
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"

#include <hip/hip_runtime.h>

namespace onnxruntime {
namespace test {

class MIGraphXExternalMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check if HIP is available
    int device_count = 0;
    hipError_t hip_err = hipGetDeviceCount(&device_count);
    if (hip_err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No HIP devices available";
    }

    // Create ORT environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MIGraphXExternalMemoryTest");
  }

  void TearDown() override {
    env_.reset();
  }

  std::unique_ptr<Ort::Env> env_;
};

// Test QueryExternalMemorySupport for HIP pointers
TEST_F(MIGraphXExternalMemoryTest, QueryHipPointerSupport) {
  Ort::SessionOptions session_options;
  
  // Use stable EP ABI - enumerate and add MIGraphX EP via V2
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "MIGraphXExecutionProvider") == 0) {
          options->AppendExecutionProvider_V2(provider_name, nullptr);
        }
      },
      &session_options));
  
  const char* model_path = "testdata/mul_1.onnx";
  Ort::Session session(*env_, model_path, session_options);
  
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  int supported = 0;
  
  // Query HIP pointer support
  OrtStatus* status = api->QueryExternalMemorySupport(
      session,
      ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HIP,
      &supported);
  
  ASSERT_EQ(status, nullptr);
  EXPECT_EQ(supported, 1) << "MIGraphX EP should support HIP pointer import";
}

// Test QueryExternalMemorySupport for unsupported types
TEST_F(MIGraphXExternalMemoryTest, QueryUnsupportedTypes) {
  Ort::SessionOptions session_options;
  
  // Use stable EP ABI
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "MIGraphXExecutionProvider") == 0) {
          options->AppendExecutionProvider_V2(provider_name, nullptr);
        }
      },
      &session_options));
  
  const char* model_path = "testdata/mul_1.onnx";
  Ort::Session session(*env_, model_path, session_options);
  
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  int supported = 0;
  
  // MIGraphX EP should NOT support D3D12 (Windows-only)
  OrtStatus* status = api->QueryExternalMemorySupport(
      session,
      ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
      &supported);
  
  ASSERT_EQ(status, nullptr);
  EXPECT_EQ(supported, 0) << "MIGraphX EP should not support D3D12 on Linux";
  
  // MIGraphX EP should NOT support CUDA pointers directly
  supported = 0;
  status = api->QueryExternalMemorySupport(
      session,
      ORT_EXTERNAL_MEMORY_HANDLE_TYPE_CUDA,
      &supported);
  
  ASSERT_EQ(status, nullptr);
  EXPECT_EQ(supported, 0) << "MIGraphX EP should not support CUDA pointers";
}

// Test HIP device pointer import
TEST_F(MIGraphXExternalMemoryTest, ImportHipDevicePointer) {
  // Allocate HIP device memory
  void* hip_ptr = nullptr;
  const size_t buffer_size = 1024 * 1024;  // 1MB
  
  hipError_t hip_err = hipMalloc(&hip_ptr, buffer_size);
  if (hip_err != hipSuccess) {
    GTEST_SKIP() << "Failed to allocate HIP memory: " << hipGetErrorString(hip_err);
  }
  
  // Create session with MIGraphX EP via stable ABI
  Ort::SessionOptions session_options;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "MIGraphXExecutionProvider") == 0) {
          options->AppendExecutionProvider_V2(provider_name, nullptr);
        }
      },
      &session_options));
  
  const char* model_path = "testdata/mul_1.onnx";
  Ort::Session session(*env_, model_path, session_options);
  
  // Create IOBinding
  Ort::IoBinding io_binding(session);
  
  // Create external memory descriptor for HIP pointer
  OrtExternalMemoryDescriptor ext_mem_desc = {};
  ext_mem_desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HIP;
  ext_mem_desc.native_handle = hip_ptr;
  ext_mem_desc.size = buffer_size;
  ext_mem_desc.offset = 0;
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
  
  // Bind HIP pointer to IOBinding
  OrtStatus* status = api->IOBindingBindExternalMemory(
      io_binding,
      "X",  // Input name
      tensor_info,
      &ext_mem_desc,
      ORT_EXTERNAL_MEMORY_ACCESS_READ_ONLY);
  
  // Should succeed
  if (status != nullptr) {
    const char* error_msg = api->GetErrorMessage(status);
    FAIL() << "Failed to bind HIP pointer to MIGraphX EP: " << error_msg;
    api->ReleaseStatus(status);
  }
  
  // Cleanup
  hipFree(hip_ptr);
}

// Test HIP pointer with different access modes
TEST_F(MIGraphXExternalMemoryTest, HipPointerAccessModes) {
  void* hip_ptr = nullptr;
  const size_t buffer_size = 1024 * sizeof(float);
  
  hipError_t hip_err = hipMalloc(&hip_ptr, buffer_size);
  ASSERT_EQ(hip_err, hipSuccess);
  
  Ort::SessionOptions session_options;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "MIGraphXExecutionProvider") == 0) {
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
    desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HIP;
    desc.native_handle = hip_ptr;
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
    desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HIP;
    desc.native_handle = hip_ptr;
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
    desc.handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HIP;
    desc.native_handle = hip_ptr;
    desc.size = buffer_size;
    desc.access_mode = ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE;
    
    OrtStatus* status = api->IOBindingBindExternalMemory(
        io_binding, "Z", tensor_info, &desc, ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE);
    
    EXPECT_EQ(status, nullptr);
    if (status) api->ReleaseStatus(status);
  }
  
  hipFree(hip_ptr);
}

// Test error cases
TEST_F(MIGraphXExternalMemoryTest, ErrorCases) {
  Ort::SessionOptions session_options;
  Ort::ThrowOnError(OrtGetApiBase()->GetApi(ORT_API_VERSION)->EnumerateAvailableProviders(
      [](const char* provider_name, void* user_data) {
        auto* options = static_cast<Ort::SessionOptions*>(user_data);
        if (strcmp(provider_name, "MIGraphXExecutionProvider") == 0) {
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
  
  // Test invalid handle type
  {
    void* hip_ptr = nullptr;
    hipMalloc(&hip_ptr, 1024);
    
    OrtExternalMemoryDescriptor desc = {};
    desc.handle_type = static_cast<OrtExternalMemoryHandleType>(999);  // Invalid
    desc.native_handle = hip_ptr;
    desc.size = 1024;
    
    const int64_t shape[] = {256};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto type_info = Ort::TypeInfo::CreateTensor(mem_info, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, shape, 1);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    
    OrtStatus* status = api->IOBindingBindExternalMemory(
        io_binding, "X", tensor_info, &desc, ORT_EXTERNAL_MEMORY_ACCESS_READ_ONLY);
    
    EXPECT_NE(status, nullptr) << "Should fail with invalid handle type";
    if (status) api->ReleaseStatus(status);
    
    hipFree(hip_ptr);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_MIGRAPHX
