// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

ORT_RUNTIME_CLASS(ExecutionProvider);

struct OrtWinApi;
typedef struct OrtWinApi OrtWinApi;

struct ID3D12Resource;
struct ID3D12Device;
struct ID3D12CommandQueue;

ORT_EXPORT const OrtWinApi* ORT_API_CALL OrtGetWindowsApi(_In_ const OrtApi* ort_api) NO_EXCEPTION;

struct OrtWinApi {
  /**
    * SessionGetExecutionProvider
	 * This api is used to get a handle to an OrtExecutionProvider.
    * Currently WinML uses this to talk directly to the DML EP and configure settings on it.
    */
  OrtStatus*(ORT_API_CALL* SessionGetExecutionProvider)(_In_ OrtSession* session, _In_ size_t index, _Out_ OrtExecutionProvider** provider)NO_EXCEPTION;

  /**
    * GetProviderMemoryInfo
	 * This api gets the memory info object associated with an EP.
    * 
    * WinML uses this to manage caller specified D3D12 inputs/outputs. It uses the memory info here to call DmlCreateGPUAllocationFromD3DResource.
    */
  OrtStatus*(ORT_API_CALL* GetProviderMemoryInfo)(_In_ OrtExecutionProvider* provider, OrtMemoryInfo** memory_info)NO_EXCEPTION;

  /**
    * GetProviderAllocator
	 * This api gets associated allocator used by a provider.
    * 
    * WinML uses this to create tensors, and needs to hold onto the allocator for the duration of the associated value's lifetime.
    */
  OrtStatus*(ORT_API_CALL* GetProviderAllocator)(_In_ OrtExecutionProvider* provider, OrtAllocator** allocator)NO_EXCEPTION;

  /**
    * FreeProviderAllocator
	 * This api frees an allocator.
    * 
    * WinML uses this to free the associated allocator for an ortvalue when creating tensors.
    * Internally this derefs a shared_ptr.
    */
  OrtStatus*(ORT_API_CALL* FreeProviderAllocator)(_In_ OrtAllocator* allocator)NO_EXCEPTION;

  /**
    * DmlExecutionProviderSetDefaultRoundingMode
	 * This api is used to configure the DML EP to turn on/off rounding.
    * 
 	 * WinML uses this to disable rounding during session initialization and then enables it again post initialization.
    */
  OrtStatus*(ORT_API_CALL* DmlExecutionProviderSetDefaultRoundingMode)(_In_ OrtExecutionProvider* dml_provider, _In_ bool is_enabled)NO_EXCEPTION;

  /**
    * DmlExecutionProviderFlushContext
	 * This api is used to flush the DML EP.
    * 
    * WinML communicates directly with DML to perform this as an optimization.
    */
  OrtStatus*(ORT_API_CALL* DmlExecutionProviderFlushContext)(_In_ OrtExecutionProvider* dml_provider)NO_EXCEPTION;

  /**
    * DmlExecutionProviderReleaseCompletedReferences
	 * This api is used to release completed references after first run the DML EP.
    * 
    * WinML communicates directly with DML to perform this as an optimization.
    */
  OrtStatus*(ORT_API_CALL* DmlExecutionProviderReleaseCompletedReferences)(_In_ OrtExecutionProvider* dml_provider)NO_EXCEPTION;

  /**
    * DmlCreateGPUAllocationFromD3DResource
	 * This api is used to create a DML EP input based on a user specified d3d12 resource.
    * 
    * WinML uses this as part of its Tensor apis to allow callers to specify their own D3D12 resources as inputs/outputs.
    */
  OrtStatus*(ORT_API_CALL* DmlCreateGPUAllocationFromD3DResource)(_In_ ID3D12Resource* pResource, _Out_ void** dml_resource)NO_EXCEPTION;

  /**
    * DmlFreeGPUAllocation
	 * This api is used free the DML EP input created by DmlCreateGPUAllocationFromD3DResource.
    * 
    * WinML uses this as part of its Tensor apis to allow callers to specify their own D3D12 resources as inputs/outputs.
    */
  OrtStatus*(ORT_API_CALL* DmlFreeGPUAllocation)(_In_ void* ptr)NO_EXCEPTION;

  /**
    * DmlGetD3D12ResourceFromAllocation
	 * This api is used to get the D3D12 resource when a OrtValue has been allocated by the DML EP and accessed via GetMutableTensorData.
    * 
    * WinML uses this in the image feature path to get the d3d resource and perform and tensorization on inputs directly into the allocated d3d12 resource.
    */
  OrtStatus*(ORT_API_CALL* DmlGetD3D12ResourceFromAllocation)(_In_ OrtExecutionProvider* provider, _In_ void* allocation, _Out_ ID3D12Resource** resource)NO_EXCEPTION;

  /**
    * DmlCopyTensor
	 * This api is used copy a tensor allocated by the DML EP Allocator to the CPU.
    * 
    * WinML uses this when graphs are evaluated with DML, and their outputs remain on the GPU but need to be copied back to the CPU.
    */
  OrtStatus*(ORT_API_CALL* DmlCopyTensor)(_In_ OrtExecutionProvider* provider, _In_ OrtValue* src, _In_ OrtValue* dst)NO_EXCEPTION;
};
