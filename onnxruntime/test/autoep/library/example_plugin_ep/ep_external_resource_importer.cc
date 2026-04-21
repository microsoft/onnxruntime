// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_external_resource_importer.h"

#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>
#include <chrono>

ExampleExternalResourceImporter::ExampleExternalResourceImporter(const ApiPtrs& apis)
    : OrtExternalResourceImporterImpl{}, apis_{apis} {
  ort_version_supported = ORT_API_VERSION;

  // Memory operations
  CanImportMemory = CanImportMemoryImpl;
  ImportMemory = ImportMemoryImpl;
  ReleaseMemory = ReleaseMemoryImpl;
  CreateTensorFromMemory = CreateTensorFromMemoryImpl;

  // Semaphore operations
  CanImportSemaphore = CanImportSemaphoreImpl;
  ImportSemaphore = ImportSemaphoreImpl;
  ReleaseSemaphore = ReleaseSemaphoreImpl;
  WaitSemaphore = WaitSemaphoreImpl;
  SignalSemaphore = SignalSemaphoreImpl;

  // Release
  Release = ReleaseImpl;
}

/*static*/
bool ORT_API_CALL ExampleExternalResourceImporter::CanImportMemoryImpl(
    _In_ const OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalMemoryHandleType handle_type) noexcept {
  // The example EP supports both D3D12 resource and heap handle types for testing
  return handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE ||
         handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleExternalResourceImporter::ImportMemoryImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ const OrtExternalMemoryDescriptor* desc,
    _Outptr_ OrtExternalMemoryHandle** out_handle) noexcept {
  auto& impl = *static_cast<ExampleExternalResourceImporter*>(this_ptr);

  if (desc == nullptr || out_handle == nullptr) {
    return impl.apis_.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments to ImportMemory");
  }

  *out_handle = nullptr;

  // Validate handle type
  if (!CanImportMemoryImpl(this_ptr, desc->handle_type)) {
    return impl.apis_.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED,
                                           "Unsupported external memory handle type");
  }

  // In a real implementation, you would:
  // 1. Open/import the native handle (e.g., cuImportExternalMemory for CUDA)
  // 2. Map the memory to get a device pointer
  //
  // For testing purposes, we simulate this by allocating CPU memory
  // that mirrors the size of the external allocation.

  auto* handle = new (std::nothrow) ExampleExternalMemoryHandle(*desc);
  if (handle == nullptr) {
    return impl.apis_.ort_api.CreateStatus(ORT_FAIL, "Failed to allocate external memory handle");
  }

  // Allocate simulated memory (using CPU memory for the example)
  size_t effective_size = desc->size_bytes - desc->offset_bytes;
  handle->simulated_ptr = std::make_unique<char[]>(effective_size);

  *out_handle = handle;
  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleExternalResourceImporter::ReleaseMemoryImpl(
    _In_ OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalMemoryHandle* handle) noexcept {
  if (handle == nullptr) {
    return;
  }

  auto* mem_handle = static_cast<ExampleExternalMemoryHandle*>(handle);
  delete mem_handle;  // destructor frees simulated_ptr
}

/*static*/
OrtStatus* ORT_API_CALL ExampleExternalResourceImporter::CreateTensorFromMemoryImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ const OrtExternalMemoryHandle* mem_handle,
    _In_ const OrtExternalTensorDescriptor* tensor_desc,
    _Outptr_ OrtValue** out_tensor) noexcept {
  auto& impl = *static_cast<ExampleExternalResourceImporter*>(this_ptr);

  if (mem_handle == nullptr || tensor_desc == nullptr || out_tensor == nullptr) {
    return impl.apis_.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments to CreateTensorFromMemory");
  }

  *out_tensor = nullptr;

  auto* handle = static_cast<const ExampleExternalMemoryHandle*>(mem_handle);

  // Calculate the data pointer with tensor offset
  void* data_ptr = handle->simulated_ptr.get() + tensor_desc->offset_bytes;

  // For the example EP, we use CPU memory info since we're simulating with CPU memory
  // In a real implementation, you would use the appropriate GPU memory info
  OrtMemoryInfo* memory_info = nullptr;
  OrtStatus* status = impl.apis_.ort_api.CreateMemoryInfo(
      "Cpu",  // For testing, we use CPU memory
      OrtDeviceAllocator,
      0,  // device ID
      OrtMemTypeDefault,
      &memory_info);

  if (status != nullptr) {
    return status;
  }

  // Calculate buffer size
  // NOTE: This is a simplified calculation for testing. Production code should:
  //   1. Calculate actual tensor size from shape + element_type
  //   2. Validate it fits within available memory region
  //   3. Use that validated size rather than subtracting offsets
  size_t buffer_size = handle->descriptor.size_bytes - handle->descriptor.offset_bytes - tensor_desc->offset_bytes;

  // Create tensor with pre-allocated memory
  status = impl.apis_.ort_api.CreateTensorWithDataAsOrtValue(
      memory_info,
      data_ptr,
      buffer_size,
      tensor_desc->shape,
      tensor_desc->rank,
      tensor_desc->element_type,
      out_tensor);

  impl.apis_.ort_api.ReleaseMemoryInfo(memory_info);
  return status;
}

/*static*/
bool ORT_API_CALL ExampleExternalResourceImporter::CanImportSemaphoreImpl(
    _In_ const OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalSemaphoreType type) noexcept {
  // The example EP supports D3D12 fence for testing
  return type == ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleExternalResourceImporter::ImportSemaphoreImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ const OrtExternalSemaphoreDescriptor* desc,
    _Outptr_ OrtExternalSemaphoreHandle** out_handle) noexcept {
  auto& impl = *static_cast<ExampleExternalResourceImporter*>(this_ptr);

  if (desc == nullptr || out_handle == nullptr) {
    return impl.apis_.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments to ImportSemaphore");
  }

  *out_handle = nullptr;

  // Validate semaphore type
  if (!CanImportSemaphoreImpl(this_ptr, desc->type)) {
    return impl.apis_.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED,
                                           "Unsupported external semaphore type");
  }

  // In a real implementation, you would:
  // 1. Import the native fence handle (e.g., cuImportExternalSemaphore for CUDA)
  //
  // For testing purposes, we create a simulated semaphore using an atomic counter

  auto* handle = new (std::nothrow) ExampleExternalSemaphoreHandle(*desc);
  if (handle == nullptr) {
    return impl.apis_.ort_api.CreateStatus(ORT_FAIL, "Failed to allocate external semaphore handle");
  }

  handle->value.store(0);

  *out_handle = handle;
  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleExternalResourceImporter::ReleaseSemaphoreImpl(
    _In_ OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalSemaphoreHandle* handle) noexcept {
  if (handle == nullptr) {
    return;
  }

  auto* sem_handle = static_cast<ExampleExternalSemaphoreHandle*>(handle);
  delete sem_handle;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleExternalResourceImporter::WaitSemaphoreImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ OrtExternalSemaphoreHandle* handle,
    _In_ OrtSyncStream* stream,
    _In_ uint64_t value) noexcept {
  auto& impl = *static_cast<ExampleExternalResourceImporter*>(this_ptr);

  if (handle == nullptr) {
    return impl.apis_.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments to WaitSemaphore");
  }

  // stream can be nullptr for synchronous wait
  (void)stream;

  auto* sem_handle = static_cast<ExampleExternalSemaphoreHandle*>(handle);

  // In a real implementation, you would:
  // 1. Queue a wait operation on the GPU stream (e.g., cuWaitExternalSemaphoresAsync)
  //
  // For testing, we do a simple spin-wait on the atomic counter
  // with a reasonable timeout to prevent infinite loops in tests

  const int max_iterations = 10000;
  int iterations = 0;
  while (sem_handle->value.load() < value && iterations < max_iterations) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    ++iterations;
  }

  if (iterations >= max_iterations) {
    return impl.apis_.ort_api.CreateStatus(ORT_FAIL, "WaitSemaphore timed out");
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL ExampleExternalResourceImporter::SignalSemaphoreImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ OrtExternalSemaphoreHandle* handle,
    _In_ OrtSyncStream* stream,
    _In_ uint64_t value) noexcept {
  auto& impl = *static_cast<ExampleExternalResourceImporter*>(this_ptr);

  if (handle == nullptr) {
    return impl.apis_.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid arguments to SignalSemaphore");
  }

  // stream can be nullptr for synchronous signal
  (void)stream;

  auto* sem_handle = static_cast<ExampleExternalSemaphoreHandle*>(handle);

  // In a real implementation, you would:
  // 1. Queue a signal operation on the GPU stream (e.g., cuSignalExternalSemaphoresAsync)
  //
  // For testing, we simply update the atomic counter

  sem_handle->value.store(value);

  return nullptr;
}

/*static*/
void ORT_API_CALL ExampleExternalResourceImporter::ReleaseImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr) noexcept {
  delete static_cast<ExampleExternalResourceImporter*>(this_ptr);
}
