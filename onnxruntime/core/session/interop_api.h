// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

namespace OrtInteropAPI {

// implementation that returns the API struct
ORT_API(const OrtInteropApi*, GetInteropApi);

ORT_API_STATUS_IMPL(CreateExternalResourceImporterForDevice, _In_ const OrtEpDevice* ep_device,
                    _Outptr_result_maybenull_ OrtExternalResourceImporter** out_importer);

ORT_API(void, ReleaseExternalResourceImporter, _Frees_ptr_opt_ OrtExternalResourceImporter* importer);

ORT_API_STATUS_IMPL(CanImportMemory, _In_ const OrtExternalResourceImporter* importer,
                    _In_ OrtExternalMemoryHandleType handle_type,
                    _Out_ bool* out_supported);

ORT_API_STATUS_IMPL(ImportMemory, _In_ OrtExternalResourceImporter* importer,
                    _In_ const OrtExternalMemoryDescriptor* desc,
                    _Outptr_ OrtExternalMemoryHandle** out_handle);

ORT_API(void, ReleaseExternalMemoryHandle, _Frees_ptr_opt_ OrtExternalMemoryHandle* handle);

ORT_API_STATUS_IMPL(CreateTensorFromMemory, _In_ OrtExternalResourceImporter* importer,
                    _In_ const OrtExternalMemoryHandle* mem_handle,
                    _In_ const OrtExternalTensorDescriptor* tensor_desc,
                    _Outptr_ OrtValue** out_tensor);

ORT_API_STATUS_IMPL(CanImportSemaphore, _In_ const OrtExternalResourceImporter* importer,
                    _In_ OrtExternalSemaphoreType type,
                    _Out_ bool* out_supported);

ORT_API_STATUS_IMPL(ImportSemaphore, _In_ OrtExternalResourceImporter* importer,
                    _In_ const OrtExternalSemaphoreDescriptor* desc,
                    _Outptr_ OrtExternalSemaphoreHandle** out_handle);

ORT_API(void, ReleaseExternalSemaphoreHandle, _Frees_ptr_opt_ OrtExternalSemaphoreHandle* handle);

ORT_API_STATUS_IMPL(WaitSemaphore, _In_ OrtExternalResourceImporter* importer,
                    _In_ OrtExternalSemaphoreHandle* semaphore_handle,
                    _In_ OrtSyncStream* stream,
                    _In_ uint64_t value);

ORT_API_STATUS_IMPL(SignalSemaphore, _In_ OrtExternalResourceImporter* importer,
                    _In_ OrtExternalSemaphoreHandle* semaphore_handle,
                    _In_ OrtSyncStream* stream,
                    _In_ uint64_t value);

}  // namespace OrtInteropAPI
