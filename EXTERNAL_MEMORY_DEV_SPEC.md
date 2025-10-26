# External Memory Import API - Development Specification

## Executive Summary

Zero-copy external memory import API for ONNX Runtime, enabling GPU memory sharing between DirectX 12 and GPU execution providers (CUDA, HIP/ROCm). Eliminates CPU staging overhead by allowing direct import of D3D12 resources into execution providers.

**Use Case**: D3D12 rendering applications performing ML inference on GPU textures/buffers.

**Performance**: Eliminates 2x memory copies (GPU→CPU→GPU), reducing latency by ~4-5ms per 1080p frame.

## 1. Problem Statement

ONNX Runtime execution providers operate on their own GPU memory. D3D12 applications must currently:

1. Download from D3D12 GPU memory to CPU
2. Upload from CPU to EP GPU memory  
3. Run inference
4. Download results to CPU
5. Upload back to D3D12

**Cost**: 2x unnecessary copies (GPU→CPU, CPU→GPU), ~4-5ms latency for 1080p RGBA texture.

**Solution**: Import D3D12 resources directly into CUDA/HIP using platform external memory APIs (`cuImportExternalMemory`, `hipImportExternalMemory`).

## 2. API Design

### 2.1 Handle Types

```c
typedef enum OrtExternalMemoryHandleType {
  ORT_EXTERNAL_MEMORY_HANDLE_TYPE_NONE = 0,
  ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 1,  // ID3D12Resource*
  ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 2,      // ID3D12Heap*
  ORT_EXTERNAL_MEMORY_HANDLE_TYPE_CUDA = 3,            // CUDA device pointer
  ORT_EXTERNAL_MEMORY_HANDLE_TYPE_HIP = 4,             // HIP device pointer
} OrtExternalMemoryHandleType;
```

Modeled after Vulkan `VkExternalMemoryHandleTypeFlagBits`. Separate D3D12_RESOURCE/D3D12_HEAP matches D3D12 allocation model.

### 2.2 Access Mode

```c
typedef enum OrtExternalMemoryAccessMode {
  ORT_EXTERNAL_MEMORY_ACCESS_READ_WRITE = 0,  // Full sync (wait + signal)
  ORT_EXTERNAL_MEMORY_ACCESS_READ_ONLY = 1,   // Wait only (input)
  ORT_EXTERNAL_MEMORY_ACCESS_WRITE_ONLY = 2,  // Signal only (output)
} OrtExternalMemoryAccessMode;
```

Enables synchronization optimization:
- `READ_ONLY`: Skip signal semaphore (D3D12 render → ORT inference)
- `WRITE_ONLY`: Skip wait semaphore (ORT generates → D3D12 renders)

### 2.3 Memory Descriptor

```c
typedef struct OrtExternalMemoryDescriptor {
  OrtExternalMemoryHandleType handle_type;
  void* native_handle;                     // Opaque: ID3D12Resource*, HANDLE, device ptr
  size_t size;
  size_t offset;                           // For D3D12_HEAP placed resources
  uint64_t flags;                          // Reserved
  OrtExternalMemoryAccessMode access_mode;
  void* wait_semaphore_handle;             // D3D12 fence shared HANDLE
  uint64_t wait_semaphore_value;           // Timeline value to wait for
  void* signal_semaphore_handle;           // D3D12 fence shared HANDLE
  uint64_t signal_semaphore_value;         // Timeline value to signal
} OrtExternalMemoryDescriptor;
```

**Key Points**:
- `native_handle` is opaque to avoid D3D12 types in public header
- Timeline semaphores enable multi-queue ordering without fence proliferation

### 2.4 Public API Functions

#### QueryExternalMemorySupport
```c
ORT_API2_STATUS(QueryExternalMemorySupport,
                _In_ const OrtSession* session,
                OrtExternalMemoryHandleType handle_type,
                _Out_ int* out_supported);
```
Returns 1 if any EP in the session supports the handle type, 0 otherwise.

#### CreateTensorFromExternalMemory
```c
ORT_API2_STATUS(CreateTensorFromExternalMemory,
                _In_ const OrtTensorTypeAndShapeInfo* info,
                _In_ const OrtExternalMemoryDescriptor* external_mem_desc,
                _In_ OrtAllocator* allocator,
                _Outptr_ OrtValue** out);
```
**Status**: Returns `NOT_IMPLEMENTED` (requires allocator→factory mapping infrastructure). Use `IOBindingBindExternalMemory` instead.

#### IOBindingBindExternalMemory
```c
ORT_API2_STATUS(IOBindingBindExternalMemory,
                _Inout_ OrtIoBinding* binding,
                _In_ const char* name,
                _In_ const OrtTensorTypeAndShapeInfo* info,
                _In_ const OrtExternalMemoryDescriptor* external_mem_desc,
                OrtExternalMemoryAccessMode access_mode);
```
Binds external memory to IOBinding input/output. **Primary API for external memory usage.**

## 3. Execution Provider Interface

Two new virtual methods in `IExecutionProvider`:

```cpp
virtual bool CanImportExternalMemory(OrtExternalMemoryHandleType handle_type) const {
  return false;  // Default: not supported
}

virtual Status ImportExternalMemory(
    const OrtExternalMemoryDescriptor& mem_desc,
    const TensorShape& shape,
    ONNXTensorElementDataType element_type,
    OrtValue& out_tensor) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "...");
}
```

Stable EP ABI support via `OrtEpFactory`:
```c
bool (*CanImportExternalMemory)(const OrtEpFactory*, const OrtMemoryDevice*, 
                                OrtExternalMemoryHandleType);
OrtStatus* (*ImportExternalMemory)(OrtEpFactory*, const OrtMemoryDevice*,
                                   const OrtExternalMemoryDescriptor*, void** device_ptr);
void (*ReleaseExternalMemory)(OrtEpFactory*, const OrtMemoryDevice*, void* device_ptr);
```

## 4. CUDA/HIP Implementation

### 4.1 NvTensorRtRtx EP (CUDA)

**Platform**: Windows only (`#ifdef _WIN32`)  
**Supported**: `D3D12_RESOURCE`, `D3D12_HEAP`

**Implementation**:
1. Import D3D12 resource: `cuImportExternalMemory(&cuda_ext_mem, &ext_mem_desc)`
2. Map to device pointer: `cuExternalMemoryGetMappedBuffer(&device_ptr, cuda_ext_mem, &buf_desc)`
3. Import semaphores: `cuImportExternalSemaphore(&semaphore, &sem_desc)`
4. Wait (if reading): `cuWaitExternalSemaphoresAsync(..., stream)`
5. Create tensor wrapping device pointer
6. Signal (if writing): `cuSignalExternalSemaphoresAsync(..., stream)`

**CUDA Types**:
- `CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE`
- `CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE`

### 4.2 MIGraphX EP (HIP/ROCm)

**Platform**: Windows only  
**Supported**: `D3D12_RESOURCE`, `D3D12_HEAP`, `HIP`

Identical to CUDA but using HIP APIs:
- `hipImportExternalMemory`
- `hipExternalMemoryGetMappedBuffer`
- `hipImportExternalSemaphore`
- `hipWaitExternalSemaphoresAsync`
- `hipSignalExternalSemaphoresAsync`

## 5. Synchronization: Timeline Semaphores

D3D12 fences and CUDA/HIP external semaphores use timeline values (monotonically increasing integers).

**D3D12 renders**:
```cpp
d3d_command_queue->Signal(fence, 1);  // Signal value 1 after render
```

**ORT imports and waits**:
```cpp
OrtExternalMemoryDescriptor desc = {
    .wait_semaphore_handle = fence_shared_handle,
    .wait_semaphore_value = 1,  // Wait for render
    .signal_semaphore_handle = fence_shared_handle,
    .signal_semaphore_value = 2,  // Signal after inference
};
// CUDA: cuWaitExternalSemaphoresAsync(fence, 1) → inference → cuSignalExternalSemaphoresAsync(fence, 2)
```

**D3D12 consumes result**:
```cpp
d3d_command_queue->Wait(fence, 2);  // Wait for inference
```

Timeline semaphores enable multi-queue ordering without creating new fence objects per operation.

## 6. Platform and Build

**Platform**: Windows only (D3D12 interop). Linux/macOS return `NOT_IMPLEMENTED`.

**Requirements**:
- Windows 10 1809+ (D3D12 shared fences)
- CUDA 10.0+ or ROCm 4.0+ (external memory APIs)
- D3D12 device and CUDA/HIP device on same physical GPU

**Dependencies**: None. Uses existing Windows SDK (d3d12.h) and CUDA/HIP headers.

**Conditional Compilation**: All D3D12 code guarded by `#ifdef _WIN32`.

## 7. Testing

**Test Files** (~1300 lines total):
- `test/autoep/test_external_memory.cc` - API surface validation
- `test/providers/cuda/test_cuda_external_memory.cc` - CUDA EP tests
- `test/providers/migraphx/test_migraphx_external_memory.cc` - HIP EP tests  
- `test/providers/nv/test_nv_external_memory.cc` - NV EP tests

**Coverage**:
- Query support for all handle types
- Import D3D12 resources with shared handles
- Access mode testing (READ_ONLY, WRITE_ONLY, READ_WRITE)
- Error cases (NULL parameters, invalid handles)
- IOBinding workflow end-to-end

## 8. Known Limitations

1. **Memory Lifecycle**: Simplified implementation leaks `CUexternalMemory`/`hipExternalMemory_t` handles. Production requires tracking and cleanup in tensor deleter.

2. **Shared Handle Ownership**: Caller must manage D3D12 `CreateSharedHandle` lifetime. HANDLE must remain valid until tensor destruction.

3. **Device Matching**: No validation that D3D12 and CUDA/HIP devices are same physical GPU. Runtime error if mismatched.

4. **Texture Layout**: No handling of D3D12 row pitch vs tensor strides. Assumes tightly packed data (works for buffers, may fail for textures).

5. **CreateTensorFromExternalMemory**: Returns `NOT_IMPLEMENTED`. Requires allocator→factory mapping infrastructure. Use `IOBindingBindExternalMemory`.

## 9. Usage Example

```c
// 1. Create D3D12 resource and fence
ID3D12Resource* texture;
ID3D12Fence* fence;
d3d12_device->CreateCommittedResource(..., &texture);
d3d12_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence));

HANDLE fence_handle;
d3d12_device->CreateSharedHandle(fence, NULL, GENERIC_ALL, NULL, &fence_handle);

// 2. Render and signal
d3d_command_list->DrawIndexed(...);
d3d_command_queue->Signal(fence, 1);

// 3. Create ORT session with CUDA EP
OrtSession* session;
api->CreateSessionOptions(&options);
api->SessionOptionsAppendExecutionProvider_CUDA(options, 0);
api->CreateSession(env, model_path, options, &session);

// 4. Bind external memory to IOBinding
OrtIoBinding* io_binding;
api->CreateIoBinding(session, &io_binding);

OrtExternalMemoryDescriptor desc = {
    .handle_type = ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
    .native_handle = texture,
    .size = 1920 * 1080 * 4 * sizeof(float),
    .access_mode = ORT_EXTERNAL_MEMORY_ACCESS_READ_ONLY,
    .wait_semaphore_handle = fence_handle,
    .wait_semaphore_value = 1,  // Wait for render
};

const int64_t shape[] = {1, 1080, 1920, 4};
api->IOBindingBindExternalMemory(io_binding, "input", tensor_info, &desc,
                                 ORT_EXTERNAL_MEMORY_ACCESS_READ_ONLY);

// 5. Run inference (CUDA waits for D3D12)
api->RunWithBinding(session, NULL, io_binding);

// 6. Cleanup
CloseHandle(fence_handle);
api->ReleaseIoBinding(io_binding);
```
