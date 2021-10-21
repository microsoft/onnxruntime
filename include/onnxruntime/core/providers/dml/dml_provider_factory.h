// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma warning(push)
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#include <d3d12.h>
#pragma warning(pop)

#ifdef __cplusplus
  #include <DirectML.h>
#else
  struct IDMLDevice;
  typedef struct IDMLDevice IDMLDevice;
#endif

// Windows pollutes the macro space, causing a build break in constants.h.
#undef OPTIONAL

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Creates a DirectML Execution Provider which executes on the hardware adapter with the given device_id, also known as
 * the adapter index. The device ID corresponds to the enumeration order of hardware adapters as given by 
 * IDXGIFactory::EnumAdapters. A device_id of 0 always corresponds to the default adapter, which is typically the 
 * primary display GPU installed on the system. A negative device_id is invalid.
*/
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_DML, _In_ OrtSessionOptions* options, int device_id);

/**
 * Creates a DirectML Execution Provider using the given DirectML device, and which executes work on the supplied D3D12
 * command queue. The DirectML device and D3D12 command queue must have the same parent ID3D12Device, or an error will
 * be returned. The D3D12 command queue must be of type DIRECT or COMPUTE (see D3D12_COMMAND_LIST_TYPE). If this 
 * function succeeds, the inference session maintains a strong reference on both the dml_device and the command_queue 
 * objects.
 * See also: DMLCreateDevice
 * See also: ID3D12Device::CreateCommandQueue
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProviderEx_DML, _In_ OrtSessionOptions* options,
               _In_ IDMLDevice* dml_device, _In_ ID3D12CommandQueue* cmd_queue);

#ifdef __cplusplus
}
#endif