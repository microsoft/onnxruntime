// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32

#include "core/providers/migraphx/migraphx_external_resource_importer.h"
#include "core/providers/migraphx/migraphx_call.h"
#include "core/providers/shared_library/provider_api.h"

#include <new>
#include <sstream>
#include <string>

namespace onnxruntime {

// ============================================================================
// MigraphxExternalMemoryHandle Implementation
// ============================================================================

MigraphxExternalMemoryHandle::MigraphxExternalMemoryHandle()
    : ext_memory(nullptr), mapped_ptr(nullptr), is_dedicated(true) {
  // Initialize base struct fields
  version = ORT_API_VERSION;
  ep_device = nullptr;
  handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
  size_bytes = 0;
  offset_bytes = 0;
  Release = ReleaseCallback;
}

void ORT_API_CALL MigraphxExternalMemoryHandle::ReleaseCallback(
    _In_ OrtExternalMemoryHandle* handle) noexcept {
  if (handle == nullptr) return;
  auto* derived = static_cast<MigraphxExternalMemoryHandle*>(handle);
  // Destroy the external memory object
  if (derived->ext_memory != nullptr) {
    [[maybe_unused]] hipError_t err = hipDestroyExternalMemory(derived->ext_memory);
  }
  delete derived;
}

// ============================================================================
// MigraphxExternalSemaphoreHandle Implementation
// ============================================================================

MigraphxExternalSemaphoreHandle::MigraphxExternalSemaphoreHandle()
    : ext_semaphore(nullptr) {
  // Initialize base struct fields
  version = ORT_API_VERSION;
  ep_device = nullptr;
  type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
  Release = ReleaseCallback;
}

void ORT_API_CALL MigraphxExternalSemaphoreHandle::ReleaseCallback(
    _In_ OrtExternalSemaphoreHandle* handle) noexcept {
  if (handle == nullptr) return;
  auto* derived = static_cast<MigraphxExternalSemaphoreHandle*>(handle);
  // Destroy the external semaphore object
  if (derived->ext_semaphore != nullptr) {
    [[maybe_unused]] hipError_t err = hipDestroyExternalSemaphore(derived->ext_semaphore);
  }
  delete derived;
}

// ============================================================================
// MigraphxExternalResourceImporterImpl Implementation
// ============================================================================

MigraphxExternalResourceImporterImpl::MigraphxExternalResourceImporterImpl(
    int device_id, const OrtApi& ort_api_in)
    : device_id_{device_id}, ort_api{ort_api_in}, ep_api{*ort_api_in.GetEpApi()} {
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

bool ORT_API_CALL MigraphxExternalResourceImporterImpl::CanImportMemoryImpl(
    _In_ const OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalMemoryHandleType handle_type) noexcept {
  // Support D3D12 resource and heap handles
  return handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE ||
         handle_type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP;
}

OrtStatus* ORT_API_CALL MigraphxExternalResourceImporterImpl::ImportMemoryImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ const OrtExternalMemoryDescriptor* desc,
    _Outptr_ OrtExternalMemoryHandle** out_handle) noexcept {
  auto& impl = *static_cast<MigraphxExternalResourceImporterImpl*>(this_ptr);

  if (desc == nullptr || out_handle == nullptr) {
    return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                     "desc and out_handle cannot be nullptr");
  }

  *out_handle = nullptr;

  // Validate handle type
  if (!CanImportMemoryImpl(this_ptr, desc->handle_type)) {
    return impl.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED,
                                     "Unsupported external memory handle type");
  }

  // Set HIP device
  hipError_t hip_err = hipSetDevice(impl.device_id_);
  if (hip_err != hipSuccess) {
    std::ostringstream oss;
    oss << "hipSetDevice failed: " << hipGetErrorString(hip_err);
    return impl.ort_api.CreateStatus(ORT_FAIL, oss.str().c_str());
  }

  // Map ORT handle type to HIP handle type
  hipExternalMemoryHandleType hip_handle_type;
  bool is_dedicated = true;

  switch (desc->handle_type) {
    case ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE:
      hip_handle_type = hipExternalMemoryHandleTypeD3D12Resource;
      is_dedicated = true;
      break;
    case ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP:
      hip_handle_type = hipExternalMemoryHandleTypeD3D12Heap;
      is_dedicated = false;
      break;
    default:
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                       "Invalid external memory handle type");
  }

  // Prepare external memory handle descriptor
  hipExternalMemoryHandleDesc mem_desc = {};
  mem_desc.type = hip_handle_type;
  mem_desc.handle.win32.handle = desc->native_handle;
  mem_desc.size = desc->size_bytes;
  mem_desc.flags = 0;

  // Import external memory
  hipExternalMemory_t ext_memory = nullptr;
  hipError_t hip_result = hipImportExternalMemory(&ext_memory, &mem_desc);
  if (hip_result != hipSuccess) {
    std::ostringstream oss;
    oss << "Failed to import external memory: " << hipGetErrorString(hip_result)
        << " (handle_type=" << desc->handle_type << ", size=" << desc->size_bytes << ")";
    return impl.ort_api.CreateStatus(ORT_FAIL, oss.str().c_str());
  }

  // Map the buffer
  hipExternalMemoryBufferDesc buffer_desc = {};
  buffer_desc.offset = desc->offset_bytes;
  buffer_desc.size = desc->size_bytes - desc->offset_bytes;
  buffer_desc.flags = 0;

  void* mapped_ptr = nullptr;
  hip_result = hipExternalMemoryGetMappedBuffer(&mapped_ptr, ext_memory, &buffer_desc);
  if (hip_result != hipSuccess) {
    [[maybe_unused]] hipError_t err = hipDestroyExternalMemory(ext_memory);
    std::ostringstream oss;
    oss << "Failed to map external memory buffer: " << hipGetErrorString(hip_result);
    return impl.ort_api.CreateStatus(ORT_FAIL, oss.str().c_str());
  }

  // Create and return the derived handle
  auto* handle = new (std::nothrow) MigraphxExternalMemoryHandle();
  if (handle == nullptr) {
    [[maybe_unused]] hipError_t err = hipDestroyExternalMemory(ext_memory);
    return impl.ort_api.CreateStatus(ORT_FAIL, "Failed to allocate external memory handle");
  }

  handle->ep_device = nullptr;
  handle->handle_type = desc->handle_type;
  handle->size_bytes = desc->size_bytes;
  handle->offset_bytes = desc->offset_bytes;
  handle->ext_memory = ext_memory;
  handle->mapped_ptr = mapped_ptr;
  handle->is_dedicated = is_dedicated;

  *out_handle = handle;
  return nullptr;
}

void ORT_API_CALL MigraphxExternalResourceImporterImpl::ReleaseMemoryImpl(
    _In_ OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalMemoryHandle* handle) noexcept {
  if (handle == nullptr) {
    return;
  }

  // The handle has a Release callback that does the actual cleanup
  auto* mem_handle = static_cast<MigraphxExternalMemoryHandle*>(handle);
  if (mem_handle->ext_memory != nullptr) {
    [[maybe_unused]] hipError_t err = hipDestroyExternalMemory(mem_handle->ext_memory);
  }
  delete mem_handle;
}

OrtStatus* ORT_API_CALL MigraphxExternalResourceImporterImpl::CreateTensorFromMemoryImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ const OrtExternalMemoryHandle* mem_handle,
    _In_ const OrtExternalTensorDescriptor* tensor_desc,
    _Outptr_ OrtValue** out_tensor) noexcept {
  auto& impl = *static_cast<MigraphxExternalResourceImporterImpl*>(this_ptr);

  if (mem_handle == nullptr || tensor_desc == nullptr || out_tensor == nullptr) {
    return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                     "mem_handle, tensor_desc, and out_tensor cannot be nullptr");
  }

  *out_tensor = nullptr;

  const auto* migraphx_handle = static_cast<const MigraphxExternalMemoryHandle*>(mem_handle);

  // Calculate tensor size
  size_t element_count = 1;
  for (size_t i = 0; i < tensor_desc->rank; ++i) {
    element_count *= static_cast<size_t>(tensor_desc->shape[i]);
  }

  size_t element_size = 0;
  switch (tensor_desc->element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      element_size = sizeof(float);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      element_size = 2;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      element_size = sizeof(int32_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      element_size = sizeof(int64_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      element_size = sizeof(uint8_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      element_size = sizeof(int8_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      element_size = sizeof(uint16_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      element_size = sizeof(int16_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      element_size = sizeof(double);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      element_size = sizeof(uint32_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      element_size = sizeof(uint64_t);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      element_size = 2;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
      element_size = 1;
      break;
    default:
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Unsupported tensor element type");
  }

  size_t tensor_size_bytes = element_count * element_size;
  size_t available_size = migraphx_handle->size_bytes - migraphx_handle->offset_bytes - tensor_desc->offset_bytes;

  if (tensor_size_bytes > available_size) {
    std::ostringstream oss;
    oss << "Tensor size (" << tensor_size_bytes << " bytes) exceeds available memory ("
        << available_size << " bytes)";
    return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
  }

  // Calculate the data pointer with offset
  void* data_ptr = static_cast<char*>(migraphx_handle->mapped_ptr) + tensor_desc->offset_bytes;

  // Create memory info for the GPU device
  OrtMemoryInfo* memory_info = nullptr;
  OrtStatus* status = impl.ort_api.CreateMemoryInfo("Hip", OrtDeviceAllocator,
                                                     impl.device_id_, OrtMemTypeDefault,
                                                     &memory_info);
  if (status != nullptr) {
    return status;
  }

  // Create tensor from the external data
  status = impl.ort_api.CreateTensorWithDataAsOrtValue(
      memory_info,
      data_ptr,
      tensor_size_bytes,
      tensor_desc->shape,
      tensor_desc->rank,
      tensor_desc->element_type,
      out_tensor);

  impl.ort_api.ReleaseMemoryInfo(memory_info);

  return status;
}

// ============================================================================
// Semaphore Operations
// ============================================================================

bool ORT_API_CALL MigraphxExternalResourceImporterImpl::CanImportSemaphoreImpl(
    _In_ const OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalSemaphoreType type) noexcept {
  // Only support D3D12 fence
  return type == ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
}

OrtStatus* ORT_API_CALL MigraphxExternalResourceImporterImpl::ImportSemaphoreImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ const OrtExternalSemaphoreDescriptor* desc,
    _Outptr_ OrtExternalSemaphoreHandle** out_handle) noexcept {
  auto& impl = *static_cast<MigraphxExternalResourceImporterImpl*>(this_ptr);

  if (desc == nullptr || out_handle == nullptr) {
    return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                     "desc and out_handle cannot be nullptr");
  }

  *out_handle = nullptr;

  // Validate semaphore type
  if (!CanImportSemaphoreImpl(this_ptr, desc->type)) {
    return impl.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED,
                                     "Unsupported external semaphore type");
  }

  // Set HIP device
  hipError_t hip_err = hipSetDevice(impl.device_id_);
  if (hip_err != hipSuccess) {
    std::ostringstream oss;
    oss << "hipSetDevice failed: " << hipGetErrorString(hip_err);
    return impl.ort_api.CreateStatus(ORT_FAIL, oss.str().c_str());
  }

  // Prepare external semaphore handle descriptor
  hipExternalSemaphoreHandleDesc sem_desc = {};
  sem_desc.type = hipExternalSemaphoreHandleTypeD3D12Fence;
  sem_desc.handle.win32.handle = desc->native_handle;
  sem_desc.flags = 0;

  // Import external semaphore
  hipExternalSemaphore_t ext_semaphore = nullptr;
  hipError_t hip_result = hipImportExternalSemaphore(&ext_semaphore, &sem_desc);
  if (hip_result != hipSuccess) {
    std::ostringstream oss;
    oss << "Failed to import external semaphore: " << hipGetErrorString(hip_result);
    return impl.ort_api.CreateStatus(ORT_FAIL, oss.str().c_str());
  }

  // Create and return the derived handle
  auto* handle = new (std::nothrow) MigraphxExternalSemaphoreHandle();
  if (handle == nullptr) {
    [[maybe_unused]] hipError_t err = hipDestroyExternalSemaphore(ext_semaphore);
    return impl.ort_api.CreateStatus(ORT_FAIL, "Failed to allocate external semaphore handle");
  }

  handle->ep_device = nullptr;
  handle->type = desc->type;
  handle->ext_semaphore = ext_semaphore;

  *out_handle = handle;
  return nullptr;
}

void ORT_API_CALL MigraphxExternalResourceImporterImpl::ReleaseSemaphoreImpl(
    _In_ OrtExternalResourceImporterImpl* /*this_ptr*/,
    _In_ OrtExternalSemaphoreHandle* handle) noexcept {
  if (handle == nullptr) {
    return;
  }

  auto* sem_handle = static_cast<MigraphxExternalSemaphoreHandle*>(handle);
  if (sem_handle->ext_semaphore != nullptr) {
    [[maybe_unused]] hipError_t err = hipDestroyExternalSemaphore(sem_handle->ext_semaphore);
  }
  delete sem_handle;
}

OrtStatus* ORT_API_CALL MigraphxExternalResourceImporterImpl::WaitSemaphoreImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ OrtExternalSemaphoreHandle* handle,
    _In_ OrtSyncStream* sync_stream,
    _In_ uint64_t value) noexcept {
  auto& impl = *static_cast<MigraphxExternalResourceImporterImpl*>(this_ptr);

  if (handle == nullptr || sync_stream == nullptr) {
    return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                     "handle and sync_stream cannot be nullptr");
  }

  auto* sem_handle = static_cast<MigraphxExternalSemaphoreHandle*>(handle);

  // Get the native HIP stream from the sync stream
  void* native_handle = impl.ort_api.SyncStream_GetHandle(sync_stream);
  if (native_handle == nullptr) {
    return impl.ort_api.CreateStatus(ORT_FAIL, "Failed to get native stream handle");
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(native_handle);

  // Set up wait parameters for D3D12 fence
  hipExternalSemaphoreWaitParams wait_params = {};
  wait_params.params.fence.value = value;
  wait_params.flags = 0;

  // Set HIP device
  hipError_t hip_err = hipSetDevice(impl.device_id_);
  if (hip_err != hipSuccess) {
    std::ostringstream oss;
    oss << "hipSetDevice failed: " << hipGetErrorString(hip_err);
    return impl.ort_api.CreateStatus(ORT_FAIL, oss.str().c_str());
  }

  // Issue async wait on the stream
  hipError_t hip_result = hipWaitExternalSemaphoresAsync(
      &sem_handle->ext_semaphore, &wait_params, 1, hip_stream);

  if (hip_result != hipSuccess) {
    std::ostringstream oss;
    oss << "Failed to wait on external semaphore: " << hipGetErrorString(hip_result)
        << " (value=" << value << ")";
    return impl.ort_api.CreateStatus(ORT_FAIL, oss.str().c_str());
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL MigraphxExternalResourceImporterImpl::SignalSemaphoreImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr,
    _In_ OrtExternalSemaphoreHandle* handle,
    _In_ OrtSyncStream* sync_stream,
    _In_ uint64_t value) noexcept {
  auto& impl = *static_cast<MigraphxExternalResourceImporterImpl*>(this_ptr);

  if (handle == nullptr || sync_stream == nullptr) {
    return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                     "handle and sync_stream cannot be nullptr");
  }

  auto* sem_handle = static_cast<MigraphxExternalSemaphoreHandle*>(handle);

  // Get the native HIP stream from the sync stream
  void* native_handle = impl.ort_api.SyncStream_GetHandle(sync_stream);
  if (native_handle == nullptr) {
    return impl.ort_api.CreateStatus(ORT_FAIL, "Failed to get native stream handle");
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(native_handle);

  // Set up signal parameters for D3D12 fence
  hipExternalSemaphoreSignalParams signal_params = {};
  signal_params.params.fence.value = value;
  signal_params.flags = 0;

  // Set HIP device
  hipError_t hip_err = hipSetDevice(impl.device_id_);
  if (hip_err != hipSuccess) {
    std::ostringstream oss;
    oss << "hipSetDevice failed: " << hipGetErrorString(hip_err);
    return impl.ort_api.CreateStatus(ORT_FAIL, oss.str().c_str());
  }

  // Issue async signal on the stream
  hipError_t hip_result = hipSignalExternalSemaphoresAsync(
      &sem_handle->ext_semaphore, &signal_params, 1, hip_stream);

  if (hip_result != hipSuccess) {
    std::ostringstream oss;
    oss << "Failed to signal external semaphore: " << hipGetErrorString(hip_result)
        << " (value=" << value << ")";
    return impl.ort_api.CreateStatus(ORT_FAIL, oss.str().c_str());
  }

  return nullptr;
}

void ORT_API_CALL MigraphxExternalResourceImporterImpl::ReleaseImpl(
    _In_ OrtExternalResourceImporterImpl* this_ptr) noexcept {
  if (this_ptr == nullptr) {
    return;
  }
  delete static_cast<MigraphxExternalResourceImporterImpl*>(this_ptr);
}

// ============================================================================
// MigraphxSyncStreamImpl Implementation
// ============================================================================

MigraphxSyncStreamImpl::MigraphxSyncStreamImpl(int device_id, const OrtApi& ort_api_in)
    : stream_(nullptr), device_id_(device_id), owns_stream_(true), ort_api{ort_api_in} {
  ort_version_supported = ORT_API_VERSION;

  // Wire up base struct function pointers
  Release = ReleaseImpl;
  GetHandle = GetHandleImpl;
  CreateNotification = nullptr;  // Not implemented for now

  // Create a HIP stream
  [[maybe_unused]] hipError_t set_result = hipSetDevice(device_id_);
  hipError_t result = hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking);
  if (result != hipSuccess) {
    stream_ = nullptr;
    owns_stream_ = false;
  }
}

MigraphxSyncStreamImpl::~MigraphxSyncStreamImpl() {
  if (owns_stream_ && stream_ != nullptr) {
    [[maybe_unused]] hipError_t err = hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void* ORT_API_CALL MigraphxSyncStreamImpl::GetHandleImpl(
    _In_ OrtSyncStreamImpl* this_ptr) noexcept {
  auto* impl = static_cast<MigraphxSyncStreamImpl*>(this_ptr);
  return static_cast<void*>(impl->stream_);
}

void ORT_API_CALL MigraphxSyncStreamImpl::ReleaseImpl(
    _In_ OrtSyncStreamImpl* this_ptr) noexcept {
  if (this_ptr == nullptr) {
    return;
  }
  delete static_cast<MigraphxSyncStreamImpl*>(this_ptr);
}

}  // namespace onnxruntime

#endif  // _WIN32
