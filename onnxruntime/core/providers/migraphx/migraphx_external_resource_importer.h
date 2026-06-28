// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _WIN32

#include "core/session/onnxruntime_c_api.h"
#include "core/providers/migraphx/migraphx_inc.h"
#include <hip/hip_runtime_api.h>

namespace onnxruntime {

/**
 * @brief Derived handle for imported external memory from D3D12 to HIP.
 *
 * Derives from OrtExternalMemoryHandle (base struct) and adds HIP-specific fields.
 * This struct holds the HIP external memory object and the mapped device pointer
 * that can be used for zero-copy tensor creation.
 */
struct MigraphxExternalMemoryHandle : OrtExternalMemoryHandle {
  hipExternalMemory_t ext_memory;  ///< HIP external memory object
  void* mapped_ptr;                ///< Mapped device pointer for tensor access
  bool is_dedicated;               ///< Whether the D3D12 resource is a dedicated allocation

  MigraphxExternalMemoryHandle();

  static void ORT_API_CALL ReleaseCallback(_In_ OrtExternalMemoryHandle* handle) noexcept;
};

/**
 * @brief Derived handle for imported external semaphore from D3D12 fence to HIP.
 *
 * Derives from OrtExternalSemaphoreHandle (base struct) and adds HIP-specific fields.
 * D3D12 timeline fences are imported as HIP external semaphores, enabling
 * GPU-GPU synchronization between D3D12 and HIP streams.
 */
struct MigraphxExternalSemaphoreHandle : OrtExternalSemaphoreHandle {
  hipExternalSemaphore_t ext_semaphore;  ///< HIP external semaphore object

  MigraphxExternalSemaphoreHandle();

  static void ORT_API_CALL ReleaseCallback(_In_ OrtExternalSemaphoreHandle* handle) noexcept;
};

/**
 * @brief Implementation of OrtExternalResourceImporterImpl for MIGraphX EP.
 *
 * This struct implements the external resource importer interface using HIP Runtime APIs
 * to import D3D12 shared resources and timeline fences for zero-copy import.
 *
 * Supported handle types:
 * - ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE -> hipExternalMemoryHandleTypeD3D12Resource
 * - ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP -> hipExternalMemoryHandleTypeD3D12Heap
 * - ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE -> hipExternalSemaphoreHandleTypeD3D12Fence
 */
struct MigraphxExternalResourceImporterImpl : OrtExternalResourceImporterImpl {
  MigraphxExternalResourceImporterImpl(int device_id, const OrtApi& ort_api_in);

  // Memory operations
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

  // Semaphore operations
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

  static void ORT_API_CALL ReleaseImpl(
      _In_ OrtExternalResourceImporterImpl* this_ptr) noexcept;

  int device_id_;
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

/**
 * @brief SyncStream implementation for MIGraphX EP.
 *
 * Wraps a hipStream_t for use in the ORT EP API stream infrastructure.
 * This enables WaitSemaphore and SignalSemaphore operations on a HIP stream.
 */
struct MigraphxSyncStreamImpl : OrtSyncStreamImpl {
  MigraphxSyncStreamImpl(int device_id, const OrtApi& ort_api_in);
  ~MigraphxSyncStreamImpl();

  static void* ORT_API_CALL GetHandleImpl(
      _In_ OrtSyncStreamImpl* this_ptr) noexcept;

  static void ORT_API_CALL ReleaseImpl(
      _In_ OrtSyncStreamImpl* this_ptr) noexcept;

  hipStream_t stream_;
  int device_id_;
  bool owns_stream_;
  const OrtApi& ort_api;
};

}  // namespace onnxruntime

#endif  // _WIN32
