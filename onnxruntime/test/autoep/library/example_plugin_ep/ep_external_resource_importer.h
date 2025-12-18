// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../plugin_ep_utils.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>

/**
 * @brief Example implementation of external memory handle.
 *
 * This mock implementation simulates imported external memory for testing purposes.
 * In a real EP, this would hold a GPU-mapped pointer from an imported D3D12/Vulkan/CUDA resource.
 */
struct ExampleExternalMemoryHandle {
  void* simulated_ptr;                      ///< Simulated mapped pointer (CPU memory for testing)
  size_t size_bytes;                        ///< Size of the imported memory
  size_t offset_bytes;                      ///< Offset into the imported memory
  OrtExternalMemoryHandleType handle_type;  ///< Original handle type
  OrtExternalMemoryAccessMode access_mode;  ///< Access mode for the imported memory

  ExampleExternalMemoryHandle()
      : simulated_ptr(nullptr), size_bytes(0), offset_bytes(0), handle_type(ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE), access_mode(ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE) {}

  ~ExampleExternalMemoryHandle() {
    // Free the simulated pointer if allocated
    if (simulated_ptr != nullptr) {
      free(simulated_ptr);
    }
  }
};

/**
 * @brief Example implementation of external semaphore handle.
 *
 * This mock implementation simulates imported external semaphores for testing purposes.
 * In a real EP, this would hold an imported D3D12 fence / Vulkan semaphore / CUDA external semaphore.
 */
struct ExampleExternalSemaphoreHandle {
  OrtExternalSemaphoreType type;  ///< Original semaphore type
  std::atomic<uint64_t> value;    ///< Simulated fence value for testing

  ExampleExternalSemaphoreHandle()
      : type(ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE), value(0) {}
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
  ExampleExternalResourceImporter(int device_id, const ApiPtrs& apis);

  // ──────────────── Memory operations ────────────────

  static bool ORT_API_CALL CanImportMemoryImpl(
      _In_ const OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalMemoryHandleType handle_type) noexcept;

  static OrtStatus* ORT_API_CALL ImportMemoryImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ const OrtExternalMemoryDescriptor* desc,
      _Outptr_ OrtExternalMemoryHandleImpl** out_handle) noexcept;

  static void ORT_API_CALL ReleaseMemoryImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalMemoryHandleImpl* handle) noexcept;

  static OrtStatus* ORT_API_CALL CreateTensorFromMemoryImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ const OrtExternalMemoryHandleImpl* mem_handle,
      _In_ const OrtExternalTensorDescriptor* tensor_desc,
      _Outptr_ OrtValue** out_tensor) noexcept;

  // ──────────────── Semaphore operations ────────────────

  static bool ORT_API_CALL CanImportSemaphoreImpl(
      _In_ const OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreType type) noexcept;

  static OrtStatus* ORT_API_CALL ImportSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ const OrtExternalSemaphoreDescriptor* desc,
      _Outptr_ OrtExternalSemaphoreHandleImpl** out_handle) noexcept;

  static void ORT_API_CALL ReleaseSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreHandleImpl* handle) noexcept;

  static OrtStatus* ORT_API_CALL WaitSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreHandleImpl* handle,
      _In_ OrtSyncStream* stream,
      _In_ uint64_t value) noexcept;

  static OrtStatus* ORT_API_CALL SignalSemaphoreImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr,
      _In_ OrtExternalSemaphoreHandleImpl* handle,
      _In_ OrtSyncStream* stream,
      _In_ uint64_t value) noexcept;

  // ──────────────── Release ────────────────

  static void ORT_API_CALL ReleaseImpl(_In_ OrtExternalResourceImporterImpl* this_ptr) noexcept;

 private:
  int device_id_;
  ApiPtrs apis_;
};
