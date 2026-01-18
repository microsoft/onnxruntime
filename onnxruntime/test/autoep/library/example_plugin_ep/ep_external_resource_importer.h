// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../plugin_ep_utils.h"

#include <atomic>
#include <cstddef>
#include <memory>

/**
 * @brief Example derived handle for imported external memory.
 *
 * Derives from OrtExternalMemoryHandle and adds example-specific fields.
 * This mock implementation simulates imported external memory for testing purposes.
 * In a real EP, this would hold a GPU-mapped pointer from an imported D3D12/Vulkan/CUDA resource.
 */
struct ExampleExternalMemoryHandle : OrtExternalMemoryHandle {
  std::unique_ptr<char[]> simulated_ptr;  ///< Simulated mapped pointer (CPU memory for testing)

  ExampleExternalMemoryHandle(const OrtExternalMemoryDescriptor& descriptor_in)
      : simulated_ptr(nullptr) {
    // Initialize base struct fields
    version = ORT_API_VERSION;
    ep_device = nullptr;
    descriptor = descriptor_in;
    Release = ReleaseCallback;
  }

  ~ExampleExternalMemoryHandle() = default;

  static void ORT_API_CALL ReleaseCallback(_In_ OrtExternalMemoryHandle* handle) noexcept {
    if (handle == nullptr) return;
    delete static_cast<ExampleExternalMemoryHandle*>(handle);
  }
};

/**
 * @brief Example derived handle for imported external semaphore.
 *
 * Derives from OrtExternalSemaphoreHandle and adds example-specific fields.
 * This mock implementation simulates imported external semaphores for testing purposes.
 * In a real EP, this would hold an imported D3D12 fence / Vulkan semaphore / CUDA external semaphore.
 */
struct ExampleExternalSemaphoreHandle : OrtExternalSemaphoreHandle {
  std::atomic<uint64_t> value;  ///< Simulated fence value for testing

  ExampleExternalSemaphoreHandle(const OrtExternalSemaphoreDescriptor& descriptor_in)
      : value(0) {
    // Initialize base struct fields
    version = ORT_API_VERSION;
    ep_device = nullptr;
    descriptor = descriptor_in;
    Release = ReleaseCallback;
  }

  static void ORT_API_CALL ReleaseCallback(_In_ OrtExternalSemaphoreHandle* handle) noexcept {
    if (handle == nullptr) return;
    delete static_cast<ExampleExternalSemaphoreHandle*>(handle);
  }
};

/**
 * @brief Example implementation of OrtExternalResourceImporterImpl.
 *
 * This is a mock implementation that simulates external resource interop for testing
 * the ORT public API without requiring actual D3D12/CUDA/Vulkan hardware.
 *
 * Key features:
 * - Reports support for D3D12 resource/heap and fence handle types
 * - Creates simulated memory mappings using CPU memory
 * - Simulates fence wait/signal operations
 * - Allows tensor creation from "imported" memory
 */
class ExampleExternalResourceImporter : public OrtExternalResourceImporterImpl {
 public:
  ExampleExternalResourceImporter(const ApiPtrs& apis);

  static bool ORT_API_CALL CanImportMemoryImpl(
      _In_ const OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalMemoryHandleType handle_type) noexcept;

  static OrtStatus* ORT_API_CALL ImportMemoryImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ const OrtExternalMemoryDescriptor* desc,
      _Outptr_ OrtExternalMemoryHandle** out_handle) noexcept;

  static void ORT_API_CALL ReleaseMemoryImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalMemoryHandle* handle) noexcept;

  static OrtStatus* ORT_API_CALL CreateTensorFromMemoryImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ const OrtExternalMemoryHandle* mem_handle,
      _In_ const OrtExternalTensorDescriptor* tensor_desc,
      _Outptr_ OrtValue** out_tensor) noexcept;

  static bool ORT_API_CALL CanImportSemaphoreImpl(
      _In_ const OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreType type) noexcept;

  static OrtStatus* ORT_API_CALL ImportSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ const OrtExternalSemaphoreDescriptor* desc,
      _Outptr_ OrtExternalSemaphoreHandle** out_handle) noexcept;

  static void ORT_API_CALL ReleaseSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreHandle* handle) noexcept;

  static OrtStatus* ORT_API_CALL WaitSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreHandle* handle,
      _In_ OrtSyncStream* stream,
      _In_ uint64_t value) noexcept;

  static OrtStatus* ORT_API_CALL SignalSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreHandle* handle,
      _In_ OrtSyncStream* stream,
      _In_ uint64_t value) noexcept;

  static void ORT_API_CALL ReleaseImpl(_In_ OrtExternalResourceImporterImpl* this_ptr) noexcept;

 private:
  ApiPtrs apis_;
};
